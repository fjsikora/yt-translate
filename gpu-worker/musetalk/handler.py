"""
Self-hosted GPU MuseTalk Lip Sync Worker Handler

Handles lip synchronization using MuseTalk.
Accepts video URL + audio URL, returns lip-synced video.
"""

import os
import sys
import time
import tempfile
import traceback
import base64
import subprocess
import yaml
from pathlib import Path
from urllib.request import urlretrieve

import self-hosted
import torch


def log(msg: str) -> None:
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_startup_info() -> None:
    """Log startup health check information."""
    log("=== MuseTalk Worker Starting ===")
    log(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        log(f"CUDA Version: {torch.version.cuda}")

    # Check MuseTalk setup
    musetalk_dir = Path("/app/MuseTalk")
    log(f"MuseTalk directory exists: {musetalk_dir.exists()}")

    models_dir = musetalk_dir / "models"
    if models_dir.exists():
        log(f"Models directory contents: {list(models_dir.glob('*'))}")

    log("=== Startup Complete ===")


def download_file(url: str, output_path: str) -> str:
    """Download file from URL."""
    log(f"Downloading from {url[:60]}...")
    urlretrieve(url, output_path)
    return output_path


def run_musetalk(
    video_path: str,
    audio_path: str,
    output_dir: str,
    use_float16: bool = True,
    bbox_shift: int = 0,
) -> str:
    """Run MuseTalk lip sync inference."""
    musetalk_dir = Path("/app/MuseTalk")

    # Create inference config
    config_path = Path(output_dir) / "inference_config.yaml"
    config_data = {
        "task_0": {
            "video_path": str(Path(video_path).absolute()),
            "audio_path": str(Path(audio_path).absolute()),
        }
    }
    if bbox_shift != 0:
        config_data["task_0"]["bbox_shift"] = bbox_shift

    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Build command
    cmd = [
        sys.executable,
        "-m", "scripts.inference",
        "--inference_config", str(config_path),
        "--result_dir", str(output_dir),
        "--gpu_id", "0",
        "--unet_config", str(musetalk_dir / "models/musetalk/musetalk.json"),
    ]

    if use_float16:
        cmd.append("--use_float16")

    log(f"Running MuseTalk inference...")
    log(f"  Video: {video_path}")
    log(f"  Audio: {audio_path}")

    # Run inference
    env = os.environ.copy()
    env.pop("MPLBACKEND", None)  # Remove Jupyter's matplotlib backend

    result = subprocess.run(
        cmd,
        cwd=str(musetalk_dir),
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minute timeout
        env=env,
    )

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown error"
        raise RuntimeError(f"MuseTalk failed:\n{error_msg}")

    # Find output file
    output_files = list(Path(output_dir).glob("*.mp4"))
    if not output_files:
        raise RuntimeError(f"No output video found in {output_dir}")

    # Return most recent mp4
    latest_output = max(output_files, key=lambda p: p.stat().st_mtime)
    log(f"Output video: {latest_output}")

    return str(latest_output)


def validate_input(job_input: dict) -> None:
    """Validate job input."""
    if "video_url" not in job_input and "video_base64" not in job_input:
        raise ValueError("Either 'video_url' or 'video_base64' is required")
    if "audio_url" not in job_input and "audio_base64" not in job_input:
        raise ValueError("Either 'audio_url' or 'audio_base64' is required")


def handler(job: dict) -> dict:
    """Main handler for MuseTalk lip sync jobs."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    start_time = time.time()
    work_dir = None

    try:
        log(f"Job {job_id} started")
        log(f"Input keys: {list(job_input.keys())}")

        validate_input(job_input)

        # Create work directory
        work_dir = tempfile.mkdtemp(prefix=f"musetalk_{job_id}_")
        log(f"Work directory: {work_dir}")

        # Get video file
        if "video_url" in job_input:
            video_path = os.path.join(work_dir, "input_video.mp4")
            download_file(job_input["video_url"], video_path)
        else:
            video_path = os.path.join(work_dir, "input_video.mp4")
            video_data = base64.b64decode(job_input["video_base64"])
            with open(video_path, "wb") as f:
                f.write(video_data)

        # Get audio file
        if "audio_url" in job_input:
            audio_path = os.path.join(work_dir, "input_audio.wav")
            download_file(job_input["audio_url"], audio_path)
        else:
            audio_path = os.path.join(work_dir, "input_audio.wav")
            audio_data = base64.b64decode(job_input["audio_base64"])
            with open(audio_path, "wb") as f:
                f.write(audio_data)

        # Get optional parameters
        use_float16 = job_input.get("use_float16", True)
        bbox_shift = job_input.get("bbox_shift", 0)

        # Run lip sync
        inference_start = time.time()
        output_path = run_musetalk(
            video_path=video_path,
            audio_path=audio_path,
            output_dir=work_dir,
            use_float16=use_float16,
            bbox_shift=bbox_shift,
        )
        inference_time = round(time.time() - inference_start, 2)

        # Read output and encode to base64
        with open(output_path, "rb") as f:
            output_base64 = base64.b64encode(f.read()).decode("utf-8")

        total_time = round(time.time() - start_time, 2)
        log(f"Job {job_id} completed in {total_time}s")

        return {
            "status": "success",
            "video_base64": output_base64,
            "metrics": {
                "inference_seconds": inference_time,
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
    except subprocess.TimeoutExpired:
        log("MuseTalk timed out")
        return {
            "status": "error",
            "error": "MuseTalk timed out after 30 minutes",
            "error_type": "timeout",
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
                import shutil
                shutil.rmtree(work_dir)
            except Exception:
                pass


if __name__ == "__main__":
    log_startup_info()
    self-hosted.serverless.start({
        "handler": handler,
        "refresh_worker": True,
    })
