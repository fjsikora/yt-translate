"""
Self-hosted GPU Pyannote Diarization Worker Handler

Handles speaker diarization using Pyannote 3.1.
Accepts audio URL or base64 audio data.
Returns speaker segments with timestamps.
"""

import os
import sys
import json
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
        log(f"CUDA Version: {torch.version.cuda}")

    try:
        import pyannote.audio
        log(f"Pyannote version: {pyannote.audio.__version__}")
    except ImportError:
        log("ERROR: pyannote.audio not installed")

    log("=== Startup Complete ===")


# Global pipeline for reuse across requests
_pipeline = None


def get_pipeline(hf_token: str):
    """Get or create the diarization pipeline."""
    global _pipeline

    if _pipeline is None:
        from pyannote.audio import Pipeline

        log("Loading Pyannote pipeline...")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        if torch.cuda.is_available():
            _pipeline.to(torch.device("cuda"))
        log("Pipeline loaded successfully")

    return _pipeline


def download_audio(url: str, output_path: str) -> str:
    """Download audio from URL."""
    log(f"Downloading audio from {url[:50]}...")
    urlretrieve(url, output_path)
    return output_path


def run_diarization(audio_path: str, hf_token: str, min_speakers: int = None, max_speakers: int = None) -> list:
    """Run Pyannote speaker diarization."""
    pipeline = get_pipeline(hf_token)

    log("Running diarization...")

    # Configure speaker count hints if provided
    diarization_params = {}
    if min_speakers is not None:
        diarization_params["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarization_params["max_speakers"] = max_speakers

    diarization = pipeline(audio_path, **diarization_params)

    # Convert to list of segments
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    return speaker_segments


def validate_input(job_input: dict) -> None:
    """Validate job input."""
    if "audio_url" not in job_input and "audio_base64" not in job_input:
        raise ValueError("Either 'audio_url' or 'audio_base64' is required")


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

        # Get HuggingFace token
        hf_token = os.environ.get("HUGGINGFACE_TOKEN") or job_input.get("hf_token")
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable or 'hf_token' input is required")

        # Get audio file
        if "audio_url" in job_input:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_path = download_audio(job_input["audio_url"], temp_file.name)
        elif "audio_base64" in job_input:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_data = base64.b64decode(job_input["audio_base64"])
            temp_file.write(audio_data)
            temp_file.flush()
            audio_path = temp_file.name
        else:
            raise ValueError("No audio source provided")

        # Optional speaker count hints
        min_speakers = job_input.get("min_speakers")
        max_speakers = job_input.get("max_speakers")

        # Run diarization
        diarization_start = time.time()
        speaker_segments = run_diarization(
            audio_path,
            hf_token,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        diarization_time = round(time.time() - diarization_start, 2)

        # Get unique speakers
        speakers = list(set(seg["speaker"] for seg in speaker_segments))

        total_time = round(time.time() - start_time, 2)
        log(f"Job {job_id} completed in {total_time}s")

        return {
            "status": "success",
            "segments": speaker_segments,
            "speakers": speakers,
            "speakers_count": len(speakers),
            "segments_count": len(speaker_segments),
            "metrics": {
                "diarization_seconds": diarization_time,
                "total_seconds": total_time,
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
            except Exception:
                pass


if __name__ == "__main__":
    log_startup_info()
    self-hosted.serverless.start({
        "handler": handler,
        "refresh_worker": True,
    })
