"""
Self-hosted GPU Serverless Handler for Video Translation.

This handler wraps the translate_video() function and handles:
- Job input validation (video_url, target_lang, voice_sample_url)
- Progress reporting via Self-hosted GPU's progress_update mechanism
- Result upload to Supabase Storage with signed URL generation
- Error handling with meaningful error messages
- Health checks and monitoring (GPU, models, memory, duration)

Usage:
    This file is the entrypoint for the Self-hosted GPU serverless worker.
    It's started automatically when the container runs.
"""

import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import self-hosted

# Add parent directory to path so we can import from root
sys.path.insert(0, str(Path(__file__).parent.parent))

from translate import translate_video, ProgressInfo
from db import (
    upload_translation_to_storage,
    get_translation_signed_url,
    TRANSLATIONS_BUCKET,
)
from config import SIGNED_URL_EXPIRATION_LONG


# ============================================================================
# Health Check and Monitoring Functions
# ============================================================================


def check_gpu_availability() -> dict[str, Any]:
    """
    Check GPU availability and return GPU information.

    Returns:
        Dict with gpu_available, gpu_count, gpu_names, cuda_version
    """
    result: dict[str, Any] = {
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "cuda_version": None,
    }

    try:
        import torch
        result["gpu_available"] = torch.cuda.is_available()
        result["gpu_count"] = torch.cuda.device_count()
        result["cuda_version"] = torch.version.cuda

        for i in range(result["gpu_count"]):
            gpu_name = torch.cuda.get_device_name(i)
            result["gpu_names"].append(gpu_name)
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Error checking GPU availability: {e}")

    return result


def check_loaded_models() -> dict[str, bool]:
    """
    Check which ML models are available/loadable.

    Returns:
        Dict with model names as keys and availability as values
    """
    models: dict[str, bool] = {
        "whisper": False,
        "demucs": False,
        "pyannote": False,
        "chatterbox": False,
        "qwen_llm": False,
    }

    # Check Whisper
    try:
        import whisper
        models["whisper"] = True
    except ImportError:
        pass

    # Check Demucs
    try:
        import demucs
        models["demucs"] = True
    except ImportError:
        pass

    # Check Pyannote
    try:
        import pyannote.audio
        models["pyannote"] = True
    except ImportError:
        pass

    # Check Chatterbox
    try:
        import chatterbox
        models["chatterbox"] = True
    except ImportError:
        pass

    # Check Transformers (for Qwen LLM)
    try:
        import transformers
        models["qwen_llm"] = True
    except ImportError:
        pass

    return models


def get_memory_usage() -> dict[str, float]:
    """
    Get current memory usage statistics.

    Returns:
        Dict with ram_used_gb, ram_total_gb, ram_percent,
        and gpu_used_gb, gpu_total_gb if GPU is available
    """
    import psutil

    result: dict[str, float] = {}

    # RAM usage
    mem = psutil.virtual_memory()
    result["ram_used_gb"] = round(mem.used / (1024**3), 2)
    result["ram_total_gb"] = round(mem.total / (1024**3), 2)
    result["ram_percent"] = round(mem.percent, 1)

    # GPU memory usage (if available)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**3)

            # Get total GPU memory
            device_props = torch.cuda.get_device_properties(0)
            gpu_total = device_props.total_memory / (1024**3)

            result["gpu_used_gb"] = round(gpu_mem_allocated, 2)
            result["gpu_reserved_gb"] = round(gpu_mem_reserved, 2)
            result["gpu_total_gb"] = round(gpu_total, 2)
    except Exception:
        pass

    return result


def log_memory_usage(stage: str) -> None:
    """Log memory usage at a specific stage of processing."""
    mem = get_memory_usage()
    ram_info = f"RAM: {mem['ram_used_gb']:.2f}/{mem['ram_total_gb']:.2f}GB ({mem['ram_percent']}%)"

    gpu_info = ""
    if "gpu_used_gb" in mem:
        gpu_info = f", GPU: {mem['gpu_used_gb']:.2f}/{mem['gpu_total_gb']:.2f}GB"

    print(f"[Memory @ {stage}] {ram_info}{gpu_info}")


def log_startup_info() -> None:
    """Log health check information on startup."""
    print("=" * 60)
    print("Self-hosted GPU Worker Starting - Health Check")
    print("=" * 60)

    # GPU Info
    gpu_info = check_gpu_availability()
    print(f"\n[GPU Status]")
    print(f"  Available: {gpu_info['gpu_available']}")
    print(f"  Count: {gpu_info['gpu_count']}")
    if gpu_info['gpu_names']:
        for i, name in enumerate(gpu_info['gpu_names']):
            print(f"  GPU {i}: {name}")
    if gpu_info['cuda_version']:
        print(f"  CUDA Version: {gpu_info['cuda_version']}")

    # Model Availability
    models = check_loaded_models()
    print(f"\n[Model Availability]")
    for model_name, available in models.items():
        status = "✓" if available else "✗"
        print(f"  {status} {model_name}")

    # Initial Memory
    mem = get_memory_usage()
    print(f"\n[Initial Memory]")
    print(f"  RAM: {mem['ram_used_gb']:.2f}/{mem['ram_total_gb']:.2f}GB ({mem['ram_percent']}%)")
    if "gpu_total_gb" in mem:
        print(f"  GPU: {mem['gpu_used_gb']:.2f}/{mem['gpu_total_gb']:.2f}GB")

    print("=" * 60)
    print("Worker ready to accept jobs")
    print("=" * 60 + "\n")


def validate_input(job_input: dict[str, Any]) -> tuple[str, str, Optional[str]]:
    """
    Validate and extract job input parameters.

    Args:
        job_input: The job input dictionary from Self-hosted GPU

    Returns:
        Tuple of (video_url, target_lang, voice_sample_url)

    Raises:
        ValueError: If required inputs are missing or invalid
    """
    # video_url is required
    video_url = job_input.get("video_url")
    if not video_url:
        raise ValueError("Missing required input: 'video_url'")

    if not isinstance(video_url, str):
        raise ValueError("'video_url' must be a string")

    # Validate it looks like a URL
    if not video_url.startswith(("http://", "https://")):
        raise ValueError("'video_url' must be a valid HTTP(S) URL")

    # target_lang is required
    target_lang = job_input.get("target_lang")
    if not target_lang:
        raise ValueError("Missing required input: 'target_lang'")

    if not isinstance(target_lang, str):
        raise ValueError("'target_lang' must be a string")

    # voice_sample_url is optional
    voice_sample_url = job_input.get("voice_sample_url")
    if voice_sample_url is not None and not isinstance(voice_sample_url, str):
        raise ValueError("'voice_sample_url' must be a string if provided")

    return video_url, target_lang, voice_sample_url


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    Self-hosted GPU serverless handler for video translation.

    This function is called for each job submitted to the Self-hosted GPU endpoint.
    It translates the video and uploads the result to Supabase Storage.

    Args:
        job: Self-hosted GPU job dictionary containing:
            - id: Unique job identifier
            - input: Dict with video_url, target_lang, voice_sample_url (optional)

    Returns:
        Dict with:
            - status: "completed" or "failed"
            - output_url: Signed URL to download the translated video (if completed)
            - duration: Video duration in seconds (if completed)
            - job_duration_seconds: Time taken to process the job
            - error: Error message (if failed)
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})

    # Start timing the job
    job_start_time = time.time()
    print(f"\n{'='*60}")
    print(f"[Job {job_id}] Starting processing")
    print(f"{'='*60}")

    # Log memory usage before processing
    log_memory_usage(f"Job {job_id} - Before processing")

    # Generate a unique work directory for this job
    work_dir = Path(f"/tmp/yt-translate-{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Validate inputs
        self-hosted.serverless.progress_update(job, "Validating inputs...")
        video_url, target_lang, voice_sample_url = validate_input(job_input)
        print(f"[Job {job_id}] Input validated - target_lang={target_lang}")

        # Create progress callback for Self-hosted GPU
        def progress_callback(info: ProgressInfo) -> None:
            """Report progress to Self-hosted GPU."""
            stage = info.get("stage", "unknown")
            progress_pct = info.get("progress_pct", 0)
            detail = info.get("detail", "")

            # Format progress message
            progress_msg = f"[{progress_pct:.0f}%] {stage}: {detail}"
            self-hosted.serverless.progress_update(job, progress_msg)

        # Run translation
        self-hosted.serverless.progress_update(job, "Starting translation pipeline...")
        result = translate_video(
            video_url=video_url,
            target_lang=target_lang,
            output_dir=work_dir,
            progress_callback=progress_callback,
        )

        # Upload result to Supabase Storage
        self-hosted.serverless.progress_update(job, "Uploading result to storage...")
        output_path = result["output_url"]  # This is actually a local path

        if not Path(output_path).exists():
            raise RuntimeError(f"Output file not found: {output_path}")

        # Generate a unique storage ID for this result
        storage_id = str(uuid.uuid4())
        storage_path = upload_translation_to_storage(storage_id, output_path)

        # Generate signed URL for download (24 hour expiry)
        # Extract the path without bucket prefix for signed URL
        path_without_bucket = storage_path.replace(f"{TRANSLATIONS_BUCKET}/", "")
        signed_url = get_translation_signed_url(storage_path, SIGNED_URL_EXPIRATION_LONG)

        self-hosted.serverless.progress_update(job, "Translation complete!")

        # Calculate job duration
        job_duration = time.time() - job_start_time

        # Log completion info
        print(f"\n[Job {job_id}] Completed successfully")
        print(f"  Duration: {job_duration:.2f}s")
        print(f"  Video duration: {result.get('duration', 0):.1f}s")
        print(f"  Segments: {result.get('segments_count', 0)}")
        print(f"  Speakers: {result.get('speakers_count', 0)}")
        log_memory_usage(f"Job {job_id} - After processing")
        print(f"{'='*60}\n")

        return {
            "status": "completed",
            "output_url": signed_url,
            "duration": result.get("duration", 0),
            "segments_count": result.get("segments_count", 0),
            "speakers_count": result.get("speakers_count", 0),
            "job_duration_seconds": round(job_duration, 2),
        }

    except ValueError as e:
        # Input validation errors - return as structured error
        job_duration = time.time() - job_start_time
        print(f"\n[Job {job_id}] Failed (validation error): {e}")
        print(f"  Duration: {job_duration:.2f}s")
        log_memory_usage(f"Job {job_id} - After failure")
        print(f"{'='*60}\n")

        return {
            "status": "failed",
            "error": str(e),
            "job_duration_seconds": round(job_duration, 2),
        }

    except Exception as e:
        # Unexpected errors - include detailed traceback for debugging
        job_duration = time.time() - job_start_time
        error_msg = str(e)
        error_trace = traceback.format_exc()

        # Log full traceback for debugging
        print(f"\n[Job {job_id}] Failed with unexpected error")
        print(f"  Duration: {job_duration:.2f}s")
        print(f"  Error: {error_msg}")
        print(f"\n{'='*40}")
        print(f"DETAILED ERROR TRACE:")
        print(f"{'='*40}")
        print(error_trace)
        print(f"{'='*40}")
        log_memory_usage(f"Job {job_id} - After failure")
        print(f"{'='*60}\n")

        return {
            "status": "failed",
            "error": f"Translation failed: {error_msg}",
            "error_trace": error_trace,
            "job_duration_seconds": round(job_duration, 2),
        }

    finally:
        # Clean up work directory
        try:
            import shutil
            if work_dir.exists():
                shutil.rmtree(work_dir)
                print(f"[Job {job_id}] Work directory cleaned up")
        except Exception as cleanup_error:
            print(f"[Job {job_id}] Warning: Failed to clean up work directory: {cleanup_error}")


# Start the Self-hosted GPU serverless worker
# refresh_worker=True ensures clean state between jobs (important for ML models)
if __name__ == "__main__":
    # Log health check information on startup
    log_startup_info()

    # Start the serverless handler
    self-hosted.serverless.start({
        "handler": handler,
        "refresh_worker": True,
    })
