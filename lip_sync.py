"""
Lip Sync Processing Module for YouTube Video Translator

Applies AI lip synchronization using MuseTalk to make video lip movements
match translated audio. Uses subprocess to call MuseTalk in its own
conda environment for dependency isolation.
"""

import subprocess
import os
import yaml
from pathlib import Path
from typing import Optional


# MuseTalk is cloned within yt-translate with its own venv
MUSETALK_DIR = Path(__file__).parent / "MuseTalk"
MUSETALK_PYTHON = MUSETALK_DIR / ".venv" / "bin" / "python"


def get_media_duration(file_path: Path) -> float:
    """
    Get duration of a video or audio file using ffprobe.

    Args:
        file_path: Path to media file

    Returns:
        Duration in seconds
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def stretch_video(
    input_path: Path,
    output_path: Path,
    target_duration: float,
    original_duration: Optional[float] = None,
) -> Path:
    """
    Stretch video to match target duration using ffmpeg setpts filter.

    This is useful when translated audio is longer than the original video.
    The video is slowed down proportionally to match the audio length.

    Args:
        input_path: Path to input video
        output_path: Path for stretched output video
        target_duration: Desired duration in seconds
        original_duration: Original video duration (will be detected if not provided)

    Returns:
        Path to the stretched video
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Video file not found: {input_path}")

    # Get original duration if not provided
    if original_duration is None:
        original_duration = get_media_duration(input_path)

    stretch_factor = target_duration / original_duration

    print(f"Stretching video...")
    print(f"  Original duration: {original_duration:.2f}s")
    print(f"  Target duration: {target_duration:.2f}s")
    print(f"  Stretch factor: {stretch_factor:.3f}x")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use setpts filter to stretch video with proper re-encoding
    # Must use explicit codec (libx264) to force full re-encoding
    # Without this, ffmpeg only changes timestamps and video freezes at original end
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"setpts={stretch_factor}*PTS",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-an",  # No audio
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg stretch failed: {result.stderr}")

    # Verify output file was created
    if not output_path.exists():
        raise RuntimeError(
            f"ffmpeg completed but output file not found: {output_path}\n"
            f"stdout: {result.stdout[-500:] if result.stdout else 'empty'}\n"
            f"stderr: {result.stderr[-500:] if result.stderr else 'empty'}"
        )

    print(f"  Output: {output_path}")
    return output_path


def apply_lip_sync(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    use_float16: bool = True,
    bbox_shift: int = 0,
    gpu_id: int = 0,
) -> Path:
    """
    Apply MuseTalk lip sync to make video match audio.

    Args:
        video_path: Path to input video file
        audio_path: Path to audio file (WAV format recommended)
        output_path: Path for output lip-synced video
        use_float16: Use FP16 for reduced VRAM (default True)
        bbox_shift: Adjust face bounding box position (default 0)
        gpu_id: GPU device ID to use (default 0)

    Returns:
        Path to the output lip-synced video

    Raises:
        FileNotFoundError: If input files don't exist
        RuntimeError: If MuseTalk processing fails
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not MUSETALK_DIR.exists():
        raise FileNotFoundError(
            f"MuseTalk not found at {MUSETALK_DIR}. "
            "Clone it with: git clone https://github.com/TMElyralab/MuseTalk.git"
        )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary inference config file (MuseTalk reads paths from YAML, not CLI args)
    config_path = output_path.parent / "inference_config.yaml"
    config_data = {
        "task_0": {
            "video_path": str(video_path.absolute()),
            "audio_path": str(audio_path.absolute()),
        }
    }
    if bbox_shift != 0:
        config_data["task_0"]["bbox_shift"] = bbox_shift

    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Build MuseTalk inference command using local venv
    cmd = [
        str(MUSETALK_PYTHON),
        "-m", "scripts.inference",
        "--inference_config", str(config_path),
        "--result_dir", str(output_path.parent.absolute()),
        "--gpu_id", str(gpu_id),
        "--unet_config", str(MUSETALK_DIR / "models/musetalk/musetalk.json"),
    ]

    if use_float16:
        cmd.append("--use_float16")

    print(f"Running MuseTalk...")
    print(f"  Video: {video_path}")
    print(f"  Audio: {audio_path}")
    print(f"  Output dir: {output_path.parent}")
    print(f"  GPU: {gpu_id}")

    try:
        # Clean environment - remove Jupyter's matplotlib backend setting
        env = os.environ.copy()
        env.pop("MPLBACKEND", None)

        result = subprocess.run(
            cmd,
            cwd=str(MUSETALK_DIR),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            env=env,
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"MuseTalk failed:\n{error_msg}")

        # MuseTalk outputs to result_dir with a generated name
        # Find the output file
        result_dir = output_path.parent
        possible_outputs = list(result_dir.glob("*.mp4"))

        if not possible_outputs:
            raise RuntimeError(
                f"MuseTalk completed but no output found in {result_dir}"
            )

        # Get the most recently created mp4
        latest_output = max(possible_outputs, key=lambda p: p.stat().st_mtime)

        # Rename to expected output path if different
        if latest_output != output_path:
            latest_output.rename(output_path)

        print(f"  Output: {output_path}")
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("MuseTalk timed out after 30 minutes")
    except FileNotFoundError as e:
        if "conda" in str(e):
            raise RuntimeError(
                "conda not found. Make sure conda is installed and in PATH."
            )
        raise


def check_musetalk_setup() -> dict:
    """
    Check if MuseTalk is properly set up.

    Returns:
        Dict with setup status and any issues found
    """
    status = {
        "musetalk_dir_exists": MUSETALK_DIR.exists(),
        "venv_exists": MUSETALK_PYTHON.exists(),
        "dependencies_ok": False,
        "issues": [],
    }

    if not status["musetalk_dir_exists"]:
        status["issues"].append(
            f"MuseTalk directory not found at {MUSETALK_DIR}"
        )
        return status

    if not status["venv_exists"]:
        status["issues"].append(
            f"MuseTalk venv not found. Create with: cd MuseTalk && uv venv --python 3.10"
        )
        return status

    # Check key dependencies
    try:
        result = subprocess.run(
            [str(MUSETALK_PYTHON), "-c",
             "import torch; import mmcv; import mmdet; import mmpose; print('ok')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        status["dependencies_ok"] = result.returncode == 0 and "ok" in result.stdout
        if not status["dependencies_ok"]:
            status["issues"].append(
                f"MuseTalk dependencies missing: {result.stderr or result.stdout}"
            )
    except subprocess.TimeoutExpired:
        status["issues"].append("Dependency check timed out")
    except Exception as e:
        status["issues"].append(f"Error checking dependencies: {e}")

    return status


if __name__ == "__main__":
    # Quick setup check when run directly
    print("Checking MuseTalk setup...")
    status = check_musetalk_setup()

    print(f"  MuseTalk directory: {'✓' if status['musetalk_dir_exists'] else '✗'}")
    print(f"  Virtual environment: {'✓' if status['venv_exists'] else '✗'}")
    print(f"  Dependencies: {'✓' if status['dependencies_ok'] else '✗'}")

    if status["issues"]:
        print("\nIssues found:")
        for issue in status["issues"]:
            print(f"  - {issue}")
    else:
        print("\n✓ MuseTalk is ready to use!")
