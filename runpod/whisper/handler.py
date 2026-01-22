"""
RunPod Whisper Worker Handler

Handles speech transcription using OpenAI Whisper.
Accepts audio URL or base64, returns transcription segments.
"""

import os
import sys
import time
import tempfile
import traceback
import base64
from pathlib import Path
from urllib.request import urlretrieve

import runpod
import torch


def log(msg: str) -> None:
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_startup_info() -> None:
    """Log startup health check information."""
    log("=== Whisper Worker Starting ===")
    log(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            log(f"  Memory: {mem_total:.1f} GB")
        log(f"CUDA Version: {torch.version.cuda}")

    # Check whisper
    try:
        import whisper
        log("Whisper imported successfully")
    except ImportError:
        log("ERROR: whisper not installed")

    log("=== Startup Complete ===")


# Global model for reuse
_model = None


def get_model(model_name: str = "large-v3"):
    """Get or create the Whisper model."""
    global _model

    if _model is None:
        import whisper

        log(f"Loading Whisper model ({model_name})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = whisper.load_model(model_name, device=device)
        log(f"Model loaded on {device}")

    return _model


def download_file(url: str, output_path: str) -> str:
    """Download file from URL."""
    log(f"Downloading from {url[:80]}...")
    urlretrieve(url, output_path)
    file_size = os.path.getsize(output_path) / 1e6
    log(f"  Downloaded {file_size:.1f} MB")
    return output_path


def run_transcription(audio_path: str, language: str = None) -> dict:
    """Run Whisper transcription on audio file."""
    model = get_model()

    log(f"Transcribing {audio_path}...")
    transcription_start = time.time()

    # Transcribe with word-level timestamps
    options = {
        "word_timestamps": True,
        "verbose": False,
    }
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)

    transcription_time = time.time() - transcription_start
    log(f"Transcription completed in {transcription_time:.1f}s")

    # Convert segments to our format
    segments = []
    for seg in result["segments"]:
        segment_data = {
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
        }
        # Include word-level timestamps if available
        if "words" in seg:
            segment_data["words"] = [
                {
                    "word": w["word"],
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                }
                for w in seg["words"]
            ]
        segments.append(segment_data)

    detected_language = result.get("language", "unknown")
    log(f"Detected language: {detected_language}")
    log(f"Found {len(segments)} segments")

    return {
        "segments": segments,
        "language": detected_language,
        "text": result["text"].strip(),
    }


def validate_input(job_input: dict) -> None:
    """Validate job input."""
    # Support both "audio" and "audio_url" for flexibility
    has_url = "audio_url" in job_input or "audio" in job_input
    has_base64 = "audio_base64" in job_input

    if not has_url and not has_base64:
        raise ValueError("Either 'audio_url', 'audio', or 'audio_base64' is required")


def handler(job: dict) -> dict:
    """Main handler for Whisper jobs."""
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

        # Get optional language hint
        language = job_input.get("language")

        # Run transcription
        transcription_start = time.time()
        result = run_transcription(temp_file.name, language=language)
        transcription_time = round(time.time() - transcription_start, 2)

        total_time = round(time.time() - start_time, 2)
        log(f"Job {job_id} completed in {total_time}s")

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "segments": result["segments"],
            "language": result["language"],
            "text": result["text"],
            "metrics": {
                "transcription_seconds": transcription_time,
                "total_seconds": total_time,
                "num_segments": len(result["segments"]),
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
    runpod.serverless.start({
        "handler": handler,
        "refresh_worker": True,
    })
