"""
RunPod Demucs Audio Separation Worker Handler

Handles audio source separation using Demucs (htdemucs model).
Accepts audio URL or base64 audio data.
Returns vocals and background audio as base64 or upload to storage.
"""

import os
import sys
import json
import time
import tempfile
import traceback
import base64
import io
from pathlib import Path
from urllib.request import urlretrieve

import runpod
import torch
import torchaudio
import soundfile as sf
import numpy as np


def log(msg: str) -> None:
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_startup_info() -> None:
    """Log startup health check information."""
    log("=== Demucs Worker Starting ===")
    log(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        log(f"CUDA Version: {torch.version.cuda}")

    try:
        import demucs
        log("Demucs imported successfully")
    except ImportError:
        log("ERROR: demucs not installed")

    log("=== Startup Complete ===")


# Global model for reuse
_model = None


def get_model():
    """Get or create the Demucs model."""
    global _model

    if _model is None:
        from demucs.pretrained import get_model

        log("Loading Demucs model (htdemucs)...")
        _model = get_model("htdemucs")
        _model.eval()
        if torch.cuda.is_available():
            _model.cuda()
        log("Model loaded successfully")

    return _model


def download_audio(url: str, output_path: str) -> str:
    """Download audio from URL."""
    log(f"Downloading audio from {url[:50]}...")
    urlretrieve(url, output_path)
    return output_path


def audio_to_base64(audio_tensor: torch.Tensor, sample_rate: int) -> str:
    """Convert audio tensor to base64 WAV string."""
    buffer = io.BytesIO()
    # Use soundfile instead of torchaudio to avoid torchcodec dependency
    # soundfile expects (samples, channels) so transpose from (channels, samples)
    audio_np = audio_tensor.numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.T  # (channels, samples) -> (samples, channels)
    sf.write(buffer, audio_np, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def run_demucs(audio_path: str) -> dict:
    """Run Demucs source separation."""
    from demucs.apply import apply_model

    model = get_model()

    log("Loading audio...")
    # Use soundfile instead of torchaudio to avoid torchcodec dependency issues
    audio_data, sr = sf.read(audio_path)
    # Convert to torch tensor and ensure correct shape (channels, samples)
    wav = torch.from_numpy(audio_data.T if audio_data.ndim > 1 else audio_data).float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # Add channel dimension for mono

    # Demucs expects stereo, convert if mono
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    # Resample if needed (Demucs expects 44100)
    if sr != 44100:
        log(f"Resampling from {sr} to 44100 Hz...")
        resampler = torchaudio.transforms.Resample(sr, 44100)
        wav = resampler(wav)
        sr = 44100

    # Add batch dimension
    wav = wav.unsqueeze(0)
    if torch.cuda.is_available():
        wav = wav.cuda()

    log("Running source separation...")
    with torch.no_grad():
        sources = apply_model(model, wav, device=wav.device)

    # Sources order: drums, bass, other, vocals
    sources = sources.squeeze(0).cpu()

    vocals = sources[3]  # vocals
    background = sources[0] + sources[1] + sources[2]  # drums + bass + other

    # Clean up GPU memory
    del wav
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "vocals": vocals,
        "background": background,
        "sample_rate": sr,
    }


def validate_input(job_input: dict) -> None:
    """Validate job input."""
    if "audio_url" not in job_input and "audio_base64" not in job_input:
        raise ValueError("Either 'audio_url' or 'audio_base64' is required")


def handler(job: dict) -> dict:
    """Main handler for Demucs jobs."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    start_time = time.time()
    temp_file = None

    try:
        log(f"Job {job_id} started")
        log(f"Input keys: {list(job_input.keys())}")

        validate_input(job_input)

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

        # Run separation
        separation_start = time.time()
        result = run_demucs(audio_path)
        separation_time = round(time.time() - separation_start, 2)

        # Convert to base64 for response
        vocals_base64 = audio_to_base64(result["vocals"], result["sample_rate"])
        background_base64 = audio_to_base64(result["background"], result["sample_rate"])

        total_time = round(time.time() - start_time, 2)
        log(f"Job {job_id} completed in {total_time}s")

        return {
            "status": "success",
            "vocals_base64": vocals_base64,
            "background_base64": background_base64,
            "sample_rate": result["sample_rate"],
            "metrics": {
                "separation_seconds": separation_time,
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
    runpod.serverless.start({
        "handler": handler,
        "refresh_worker": True,
    })
