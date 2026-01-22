"""
Self-hosted GPU MuseTalk Lip Sync Worker Handler

Handles lip synchronization using MuseTalk.
Accepts video URL + audio URL, returns lip-synced video as URL (uploaded to Self-hosted GPU storage).
"""

import os
import sys
import time
import tempfile
import traceback
import base64
import subprocess
import shutil
import yaml
from pathlib import Path
from urllib.request import urlretrieve

import self-hosted
from self-hosted.serverless.utils import rp_upload
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
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            log(f"  Memory: {mem_total:.1f} GB")
        log(f"CUDA Version: {torch.version.cuda}")

    # Check MuseTalk setup
    musetalk_dir = Path("/app/MuseTalk")
    log(f"MuseTalk directory exists: {musetalk_dir.exists()}")

    models_dir = musetalk_dir / "models"
    if models_dir.exists():
        model_subdirs = [d.name for d in models_dir.iterdir() if d.is_dir()]
        log(f"Model directories: {model_subdirs}")

        # Check for key model files
        musetalk_model = models_dir / "musetalk" / "pytorch_model.bin"
        dwpose_model = models_dir / "dwpose" / "dw-ll_ucoco_384.pth"
        whisper_model = models_dir / "whisper" / "tiny.pt"

        log(f"  musetalk model: {'OK' if musetalk_model.exists() else 'MISSING'}")
        log(f"  dwpose model: {'OK' if dwpose_model.exists() else 'MISSING'}")
        log(f"  whisper model: {'OK' if whisper_model.exists() else 'MISSING'}")

    log("=== Startup Complete ===")


# Global flag for model download status
_models_ready = False


def ensure_models_downloaded() -> None:
    """Download models on first request if not present."""
    global _models_ready
    if _models_ready:
        return

    from huggingface_hub import snapshot_download
    import urllib.request

    models_dir = Path("/app/MuseTalk/models")

    # Check for actual model weight files, not just config
    required_files = [
        models_dir / "musetalk" / "pytorch_model.bin",
        models_dir / "dwpose" / "dw-ll_ucoco_384.pth",
        models_dir / "face-parse-bisent" / "79999_iter.pth",
        models_dir / "sd-vae-ft-mse" / "diffusion_pytorch_model.safetensors",
    ]

    missing = [f for f in required_files if not f.exists()]

    if not missing:
        log("All model files present, skipping download")
        _models_ready = True
        return

    log(f"Missing model files: {[f.name for f in missing]}")

    log("Downloading models (first request, this may take 2-3 minutes)...")

    # Ensure directories exist
    for subdir in ["musetalk", "sd-vae-ft-mse", "whisper", "dwpose", "face-parse-bisent"]:
        (models_dir / subdir).mkdir(parents=True, exist_ok=True)

    # MuseTalk models from HuggingFace
    log("  Downloading MuseTalk models...")
    snapshot_download(
        "TMElyralab/MuseTalk",
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["*.json", "*.bin", "*.pth", "*.pt", "*.safetensors"]
    )

    # SD VAE
    log("  Downloading SD VAE...")
    snapshot_download(
        "stabilityai/sd-vae-ft-mse",
        local_dir=str(models_dir / "sd-vae-ft-mse"),
        local_dir_use_symlinks=False
    )

    # Whisper tiny
    log("  Downloading Whisper...")
    snapshot_download(
        "openai/whisper-tiny",
        local_dir=str(models_dir / "whisper"),
        local_dir_use_symlinks=False
    )

    # DWPose
    log("  Downloading DWPose...")
    snapshot_download(
        "yzd-v/DWPose",
        local_dir=str(models_dir / "dwpose"),
        local_dir_use_symlinks=False
    )

    # Face parsing resnet18
    face_parse_dir = models_dir / "face-parse-bisent"
    resnet_path = face_parse_dir / "resnet18-5c106cde.pth"
    if not resnet_path.exists():
        log("  Downloading resnet18...")
        urllib.request.urlretrieve(
            "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            str(resnet_path)
        )

    # Face parsing model (79999_iter.pth) via gdown
    iter_path = face_parse_dir / "79999_iter.pth"
    if not iter_path.exists():
        log("  Downloading face-parse model...")
        try:
            import gdown
            gdown.download(id="154JgKpzCPW82qINcVieuPH3fZ2e0P812", output=str(iter_path), quiet=False)
        except Exception as e:
            log(f"  Warning: gdown failed ({e}), trying direct download...")
            # Fallback: try direct Google Drive URL
            urllib.request.urlretrieve(
                "https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812",
                str(iter_path)
            )

    # Create symlinks
    sd_vae_link = models_dir / "sd-vae"
    if not sd_vae_link.exists():
        log("  Creating sd-vae symlink...")
        sd_vae_link.symlink_to(models_dir / "sd-vae-ft-mse")

    config_link = models_dir / "musetalk" / "config.json"
    if not config_link.exists():
        log("  Creating config.json symlink...")
        config_link.symlink_to(models_dir / "musetalk" / "musetalk.json")

    _models_ready = True
    log("Model download complete!")


def download_file(url: str, output_path: str) -> str:
    """Download file from URL."""
    log(f"Downloading from {url[:80]}...")
    urlretrieve(url, output_path)
    file_size = os.path.getsize(output_path) / 1e6
    log(f"  Downloaded {file_size:.1f} MB")
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

    log(f"Created inference config at {config_path}")

    # Build command
    cmd = [
        sys.executable,
        "-m", "scripts.inference",
        "--inference_config", str(config_path),
        "--result_dir", str(output_dir),
        "--gpu_id", "0",
    ]

    # Add float16 flag if requested
    if use_float16:
        cmd.append("--use_float16")

    log(f"Running MuseTalk inference...")
    log(f"  Video: {video_path}")
    log(f"  Audio: {audio_path}")
    log(f"  Output dir: {output_dir}")
    log(f"  Float16: {use_float16}")
    log(f"  BBox shift: {bbox_shift}")

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

    # Log output for debugging
    if result.stdout:
        for line in result.stdout.split("\n")[-20:]:  # Last 20 lines
            if line.strip():
                log(f"  stdout: {line}")
    if result.stderr:
        for line in result.stderr.split("\n")[-10:]:  # Last 10 lines of stderr
            if line.strip():
                log(f"  stderr: {line}")

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown error"
        raise RuntimeError(f"MuseTalk failed with code {result.returncode}:\n{error_msg[-2000:]}")

    # Find output file (search recursively, exclude input file)
    # Note: MuseTalk names output like "input_video_input_audio.mp4" so we
    # only exclude the exact input filename, not files containing "input_video"
    output_files = [
        f for f in Path(output_dir).glob("**/*.mp4")
        if f.name != "input_video.mp4"
    ]

    if not output_files:
        # Log directory contents for debugging
        all_files = list(Path(output_dir).rglob("*"))
        log(f"Directory contents: {[str(f) for f in all_files[:20]]}")
        raise RuntimeError(f"No output video found in {output_dir} (searched recursively)")

    # Return most recent mp4 (excluding input)
    latest_output = max(output_files, key=lambda p: p.stat().st_mtime)
    output_size = latest_output.stat().st_size / 1e6
    log(f"Output video: {latest_output} ({output_size:.1f} MB)")

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

        # Ensure models are downloaded (first request only)
        ensure_models_downloaded()

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
            log(f"Wrote video from base64: {len(video_data) / 1e6:.1f} MB")

        # Get audio file
        if "audio_url" in job_input:
            audio_path = os.path.join(work_dir, "input_audio.wav")
            download_file(job_input["audio_url"], audio_path)
        else:
            audio_path = os.path.join(work_dir, "input_audio.wav")
            audio_data = base64.b64decode(job_input["audio_base64"])
            with open(audio_path, "wb") as f:
                f.write(audio_data)
            log(f"Wrote audio from base64: {len(audio_data) / 1e6:.1f} MB")

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

        # Upload output video to S3-compatible storage
        output_size = os.path.getsize(output_path) / 1e6
        log(f"Uploading output video ({output_size:.1f} MB) to storage...")
        video_url = rp_upload.upload_file_to_bucket(
            file_name=f"{job_id}_lipsynced.mp4",
            file_location=output_path,
            bucket_name="self-hosted-outputs"
        )
        log(f"Uploaded video: {video_url}")

        total_time = round(time.time() - start_time, 2)
        log(f"Job {job_id} completed in {total_time}s")

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "video_url": video_url,
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
        log("MuseTalk timed out after 30 minutes")
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
                shutil.rmtree(work_dir)
                log(f"Cleaned up work directory: {work_dir}")
            except Exception as e:
                log(f"Warning: Failed to cleanup work directory: {e}")


if __name__ == "__main__":
    log_startup_info()
    self-hosted.serverless.start({
        "handler": handler,
        "refresh_worker": True,
    })
