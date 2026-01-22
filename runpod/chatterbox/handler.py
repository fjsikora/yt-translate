"""
RunPod Chatterbox Worker Handler

Handles text-to-speech using Chatterbox multilingual TTS.
Accepts text + voice sample, returns synthesized audio URL.
"""

import os
import sys
import time
import tempfile
import traceback
import base64
import shutil
from pathlib import Path
from urllib.request import urlretrieve

import runpod
from runpod.serverless.utils import rp_upload
import torch
import soundfile as sf


def log(msg: str) -> None:
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_startup_info() -> None:
    """Log startup health check information."""
    log("=== Chatterbox Worker Starting ===")
    log(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            log(f"  Memory: {mem_total:.1f} GB")
        log(f"CUDA Version: {torch.version.cuda}")

    # Check chatterbox
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        log("Chatterbox imported successfully")
    except ImportError as e:
        log(f"ERROR: chatterbox not installed: {e}")

    log("=== Startup Complete ===")


# Global model for reuse
_model = None


def get_model():
    """Get or create the Chatterbox model."""
    global _model

    if _model is None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Loading Chatterbox model on {device}...")
        _model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        log("Model loaded successfully")

    return _model


def download_file(url: str, output_path: str) -> str:
    """Download file from URL."""
    log(f"Downloading from {url[:80]}...")
    urlretrieve(url, output_path)
    file_size = os.path.getsize(output_path) / 1e6
    log(f"  Downloaded {file_size:.1f} MB")
    return output_path


def run_tts(
    text: str,
    voice_sample_path: str,
    language: str = "en",
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
) -> tuple:
    """Run Chatterbox TTS synthesis."""
    model = get_model()

    log(f"Synthesizing: '{text[:50]}...' in language '{language}'")
    synthesis_start = time.time()

    # Generate audio
    audio = model.generate(
        text=text,
        audio_prompt_path=voice_sample_path,
        language_id=language,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
    )

    synthesis_time = time.time() - synthesis_start
    log(f"Synthesis completed in {synthesis_time:.1f}s")

    # Get sample rate from model
    sample_rate = 24000  # Chatterbox default

    return audio, sample_rate


def validate_input(job_input: dict) -> None:
    """Validate job input."""
    if "text" not in job_input:
        raise ValueError("'text' is required")

    has_voice_url = "voice_sample_url" in job_input
    has_voice_base64 = "voice_sample_base64" in job_input

    if not has_voice_url and not has_voice_base64:
        raise ValueError("Either 'voice_sample_url' or 'voice_sample_base64' is required")


def handler(job: dict) -> dict:
    """Main handler for Chatterbox jobs."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    start_time = time.time()
    work_dir = None

    try:
        log(f"Job {job_id} started")
        log(f"Input keys: {list(job_input.keys())}")

        validate_input(job_input)

        # Create work directory
        work_dir = tempfile.mkdtemp(prefix=f"chatterbox_{job_id}_")

        # Get voice sample
        voice_sample_path = os.path.join(work_dir, "voice_sample.wav")
        if "voice_sample_url" in job_input:
            download_file(job_input["voice_sample_url"], voice_sample_path)
        else:
            voice_data = base64.b64decode(job_input["voice_sample_base64"])
            with open(voice_sample_path, "wb") as f:
                f.write(voice_data)
            log(f"Wrote voice sample from base64: {len(voice_data) / 1e6:.1f} MB")

        # Get parameters
        text = job_input["text"]
        language = job_input.get("language", "en")
        cfg_weight = job_input.get("cfg_weight", 0.5)
        exaggeration = job_input.get("exaggeration", 0.5)

        # Run TTS
        synthesis_start = time.time()
        audio, sample_rate = run_tts(
            text=text,
            voice_sample_path=voice_sample_path,
            language=language,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
        )
        synthesis_time = round(time.time() - synthesis_start, 2)

        # Save output audio
        output_path = os.path.join(work_dir, "output.wav")

        # Convert to numpy if tensor
        if torch.is_tensor(audio):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # Ensure correct shape for soundfile (samples,) or (samples, channels)
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        sf.write(output_path, audio_np, sample_rate)
        output_size = os.path.getsize(output_path) / 1e6
        log(f"Output audio: {output_size:.2f} MB")

        # Upload to storage
        log("Uploading output to storage...")
        audio_url = rp_upload.upload_file_to_bucket(
            file_name=f"{job_id}_tts_output.wav",
            file_location=output_path,
            bucket_name="runpod-outputs"
        )
        log(f"Uploaded: {audio_url}")

        total_time = round(time.time() - start_time, 2)
        log(f"Job {job_id} completed in {total_time}s")

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "audio_url": audio_url,
            "sample_rate": sample_rate,
            "metrics": {
                "synthesis_seconds": synthesis_time,
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
        # Cleanup work directory
        if work_dir and os.path.exists(work_dir):
            try:
                shutil.rmtree(work_dir)
                log(f"Cleaned up work directory")
            except Exception as e:
                log(f"Warning: Failed to cleanup work directory: {e}")


if __name__ == "__main__":
    log_startup_info()
    runpod.serverless.start({
        "handler": handler,
        "refresh_worker": True,
    })
