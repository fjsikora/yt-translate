"""
Self-hosted GPU Chatterbox TTS Worker Handler

Handles text-to-speech generation with voice cloning.
Accepts translated segments and voice samples, returns generated audio.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import traceback
from pathlib import Path

import self-hosted
import psutil
import torch
import torchaudio


def log(msg: str) -> None:
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_memory_usage() -> dict:
    """Get current memory usage."""
    mem = psutil.virtual_memory()
    result = {
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "ram_total_gb": round(mem.total / (1024**3), 2),
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            result[f"gpu_{i}_used_gb"] = round(allocated, 2)
            result[f"gpu_{i}_total_gb"] = round(total, 2)
    return result


def log_startup_info() -> None:
    """Log startup health check information."""
    log("=== Chatterbox TTS Worker Starting ===")
    log(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        log(f"CUDA Version: {torch.version.cuda}")

    # Check model imports
    try:
        from chatterbox.tts import ChatterboxTTS
        log("Chatterbox import: OK")
    except ImportError as e:
        log(f"Chatterbox import: FAILED - {e}")

    log(f"Memory: {get_memory_usage()}")
    log("=== Startup Complete ===")


# Global model cache
_tts_model = None


def get_tts_model():
    """Get or load the Chatterbox TTS model."""
    global _tts_model
    if _tts_model is None:
        from chatterbox.tts import ChatterboxTTS
        log("Loading Chatterbox TTS model...")
        _tts_model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
        log("Chatterbox model loaded")
    return _tts_model


def generate_speech(
    model,
    text: str,
    voice_sample_path: str,
    output_path: str,
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
) -> float:
    """Generate speech for a single segment."""
    # Load voice sample
    audio_prompt, sr = torchaudio.load(voice_sample_path)

    # Generate speech
    wav = model.generate(
        text=text,
        audio_prompt=audio_prompt,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
    )

    # Save output (Chatterbox outputs at 24kHz)
    output_sr = 24000
    torchaudio.save(output_path, wav.unsqueeze(0).cpu(), output_sr)

    # Return duration
    return wav.shape[0] / output_sr


def validate_input(job_input: dict) -> None:
    """Validate job input."""
    if "segments" not in job_input:
        raise ValueError("'segments' is required")
    if not isinstance(job_input["segments"], list):
        raise ValueError("'segments' must be a list")
    if len(job_input["segments"]) == 0:
        raise ValueError("'segments' cannot be empty")

    # Check voice samples
    if "voice_samples" not in job_input:
        raise ValueError("'voice_samples' is required (dict mapping speaker to audio URL)")


def download_file(url: str, output_path: str) -> str:
    """Download a file from URL."""
    import httpx

    with httpx.Client(timeout=60.0) as client:
        response = client.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
    return output_path


def handler(job: dict) -> dict:
    """Main handler for TTS jobs."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    start_time = time.time()
    work_dir = None

    try:
        log(f"Job {job_id} started")
        log(f"Memory before: {get_memory_usage()}")

        validate_input(job_input)

        segments = job_input["segments"]
        voice_samples = job_input["voice_samples"]  # {speaker: url}
        cfg_weight = job_input.get("cfg_weight", 0.5)
        exaggeration = job_input.get("exaggeration", 0.5)

        log(f"Processing {len(segments)} segments with {len(voice_samples)} voice samples")

        # Create work directory
        work_dir = tempfile.mkdtemp(prefix=f"tts_{job_id}_")
        log(f"Work directory: {work_dir}")

        # Download voice samples
        voice_sample_paths = {}
        for speaker, url in voice_samples.items():
            sample_path = os.path.join(work_dir, f"voice_{speaker}.wav")
            log(f"Downloading voice sample for {speaker}...")
            download_file(url, sample_path)
            voice_sample_paths[speaker] = sample_path

        # Load TTS model
        model = get_tts_model()

        # Generate speech for each segment
        generated_audio = []
        total_duration = 0.0

        for i, seg in enumerate(segments):
            seg_id = seg.get("id", i)
            text = seg["text"]
            speaker = seg.get("speaker", "SPEAKER_00")

            # Get voice sample for this speaker
            voice_path = voice_sample_paths.get(speaker)
            if not voice_path:
                # Fallback to first available voice sample
                voice_path = list(voice_sample_paths.values())[0]
                log(f"Warning: No voice sample for {speaker}, using fallback")

            output_path = os.path.join(work_dir, f"segment_{seg_id}.wav")

            log(f"Generating segment {i+1}/{len(segments)}: '{text[:50]}...'")
            duration = generate_speech(
                model=model,
                text=text,
                voice_sample_path=voice_path,
                output_path=output_path,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration,
            )

            generated_audio.append({
                "id": seg_id,
                "path": output_path,
                "duration": duration,
                "speaker": speaker,
                "original_start": seg.get("start", 0),
                "original_end": seg.get("end", 0),
            })
            total_duration += duration

            # Report progress
            progress = (i + 1) / len(segments) * 100
            self-hosted.serverless.progress_update(job, f"Generated {i+1}/{len(segments)} segments")

        metrics = {
            "total_seconds": round(time.time() - start_time, 2),
            "segments_processed": len(segments),
            "total_audio_duration": round(total_duration, 2),
        }

        log(f"Memory after: {get_memory_usage()}")
        log(f"Job {job_id} completed in {metrics['total_seconds']}s")

        return {
            "status": "success",
            "generated_audio": generated_audio,
            "total_duration": total_duration,
            "segments_count": len(segments),
            "metrics": metrics,
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
        # Note: Don't cleanup work_dir here - audio files need to be uploaded first
        # The orchestrator should handle cleanup after uploading
        pass


# Run startup checks and start handler
if __name__ == "__main__":
    log_startup_info()
    self-hosted.serverless.start({
        "handler": handler,
        "refresh_worker": True,
    })
