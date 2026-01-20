"""
Self-hosted GPU Serverless Handler for Video Translation.

This handler wraps the translate_video() function and handles:
- Job input validation (video_url, target_lang, voice_sample_url)
- Progress reporting via Self-hosted GPU's progress_update mechanism
- Result upload to Supabase Storage with signed URL generation
- Error handling with meaningful error messages

Usage:
    This file is the entrypoint for the Self-hosted GPU serverless worker.
    It's started automatically when the container runs.
"""

import os
import sys
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
            - error: Error message (if failed)
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})

    # Generate a unique work directory for this job
    work_dir = Path(f"/tmp/yt-translate-{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Validate inputs
        self-hosted.serverless.progress_update(job, "Validating inputs...")
        video_url, target_lang, voice_sample_url = validate_input(job_input)

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

        return {
            "status": "completed",
            "output_url": signed_url,
            "duration": result.get("duration", 0),
            "segments_count": result.get("segments_count", 0),
            "speakers_count": result.get("speakers_count", 0),
        }

    except ValueError as e:
        # Input validation errors - return as structured error
        return {
            "status": "failed",
            "error": str(e),
        }

    except Exception as e:
        # Unexpected errors - include traceback for debugging
        error_msg = str(e)
        error_trace = traceback.format_exc()

        # Log full traceback for debugging
        print(f"Job {job_id} failed with error: {error_msg}")
        print(f"Traceback:\n{error_trace}")

        return {
            "status": "failed",
            "error": f"Translation failed: {error_msg}",
        }

    finally:
        # Clean up work directory
        try:
            import shutil
            if work_dir.exists():
                shutil.rmtree(work_dir)
        except Exception as cleanup_error:
            print(f"Warning: Failed to clean up work directory: {cleanup_error}")


# Start the Self-hosted GPU serverless worker
# refresh_worker=True ensures clean state between jobs (important for ML models)
if __name__ == "__main__":
    self-hosted.serverless.start({
        "handler": handler,
        "refresh_worker": True,
    })
