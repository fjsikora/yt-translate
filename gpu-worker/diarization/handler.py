"""
Self-hosted GPU Diarization Worker Handler

Handles speaker diarization using pyannote.audio.
Accepts audio URL or base64, returns speaker segments.
"""

import os
import sys
import time
import tempfile
import traceback
import base64
from pathlib import Path
from urllib.request import urlretrieve

import self-hosted
import torch


def log(msg: str) -> None:
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_startup_info() -> None:
    """Log startup health check information."""
    log("=== Diarization Worker Starting ===")
    log(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            log(f"  Memory: {mem_total:.1f} GB")
        log(f"CUDA Version: {torch.version.cuda}")

    # Check pyannote
    try:
        import pyannote.audio
        log(f"pyannote.audio version: {pyannote.audio.__version__}")
    except ImportError:
        log("ERROR: pyannote.audio not installed")

    # Check HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    log(f"HuggingFace Token: {'Set' if hf_token else 'NOT SET'}")

    log("=== Startup Complete ===")


# Global pipeline for reuse
_pipeline = None


def get_pipeline():
    """Get or create the diarization pipeline."""
    global _pipeline

    if _pipeline is None:
        from pyannote.audio import Pipeline

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN or HUGGINGFACE_TOKEN environment variable required")

        log("Loading pyannote diarization pipeline...")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

        if torch.cuda.is_available():
            _pipeline.to(torch.device("cuda"))
            log("Pipeline moved to GPU")

        log("Pipeline loaded successfully")

    return _pipeline


def download_file(url: str, output_path: str) -> str:
    """Download file from URL."""
    log(f"Downloading from {url[:80]}...")
    urlretrieve(url, output_path)
    file_size = os.path.getsize(output_path) / 1e6
    log(f"  Downloaded {file_size:.1f} MB")
    return output_path


def run_diarization(audio_path: str) -> list:
    """Run speaker diarization on audio file."""
    pipeline = get_pipeline()

    log(f"Running diarization on {audio_path}...")
    diarization_start = time.time()

    # Run diarization
    diarization = pipeline(audio_path)

    diarization_time = time.time() - diarization_start
    log(f"Diarization completed in {diarization_time:.1f}s")

    # Convert to list of segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        })

    log(f"Found {len(segments)} segments with {len(set(s['speaker'] for s in segments))} speakers")

    return segments


def validate_input(job_input: dict) -> None:
    """Validate job input."""
    # Support both "audio" and "audio_url" for flexibility
    has_url = "audio_url" in job_input or "audio" in job_input
    has_base64 = "audio_base64" in job_input

    if not has_url and not has_base64:
        raise ValueError("Either 'audio_url', 'audio', or 'audio_base64' is required")


def handler(job: dict) -> dict:
    """Main handler for diarization jobs."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    start_time = time.time()
    temp_file = None

    try:
        log(f"Job {job_id} started")
        log(f"Input keys: {list(job_input.keys())}")

        validate_input(job_input)

        # Create temp file for audio
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

        # Get audio file
        if "audio_url" in job_input:
            download_file(job_input["audio_url"], temp_file.name)
        elif "audio" in job_input:
            # Support "audio" key for URL (compatibility)
            download_file(job_input["audio"], temp_file.name)
        else:
            audio_data = base64.b64decode(job_input["audio_base64"])
            with open(temp_file.name, "wb") as f:
                f.write(audio_data)
            log(f"Wrote audio from base64: {len(audio_data) / 1e6:.1f} MB")

        # Run diarization
        diarization_start = time.time()
        segments = run_diarization(temp_file.name)
        diarization_time = round(time.time() - diarization_start, 2)

        # Calculate speaker stats
        speakers = set(s["speaker"] for s in segments)
        num_speakers = len(speakers)

        total_time = round(time.time() - start_time, 2)
        log(f"Job {job_id} completed in {total_time}s")

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "segments": segments,
            "num_speakers": num_speakers,
            "metrics": {
                "diarization_seconds": diarization_time,
                "total_seconds": total_time,
                "num_segments": len(segments),
            },
        }

    except ValueError as e:
        log(f"Validation error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": "validation",
        }
    except Exception as e:
        log(f"Error: {e}")
        log(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "error_type": "runtime",
            "error_trace": traceback.format_exc(),
        }
    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                log(f"Cleaned up temp file")
            except Exception as e:
                log(f"Warning: Failed to cleanup temp file: {e}")


if __name__ == "__main__":
    log_startup_info()
    self-hosted.serverless.start({
        "handler": handler,
        "refresh_worker": True,
    })
