"""
Lip Sync Processing Module for YouTube Video Translator

Applies AI lip synchronization using MuseTalk to make video lip movements
match translated audio. Uses subprocess to call MuseTalk in its own
conda environment for dependency isolation.
"""

import subprocess
import os
from pathlib import Path
from typing import Optional


# MuseTalk is cloned within yt-translate with its own venv
MUSETALK_DIR = Path(__file__).parent / "MuseTalk"
MUSETALK_PYTHON = MUSETALK_DIR / ".venv" / "bin" / "python"


def apply_lip_sync(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    use_float16: bool = True,
    bbox_shift: int = 0,
) -> Path:
    """
    Apply MuseTalk lip sync to make video match audio.

    Args:
        video_path: Path to input video file
        audio_path: Path to audio file (WAV format recommended)
        output_path: Path for output lip-synced video
        use_float16: Use FP16 for reduced VRAM (default True)
        bbox_shift: Adjust face bounding box position (default 0)

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

    # Build MuseTalk inference command using local venv
    cmd = [
        str(MUSETALK_PYTHON),
        "-m", "scripts.inference",
        "--video_path", str(video_path.absolute()),
        "--audio_path", str(audio_path.absolute()),
        "--result_dir", str(output_path.parent.absolute()),
    ]

    if use_float16:
        cmd.append("--use_float16")

    if bbox_shift != 0:
        cmd.extend(["--bbox_shift", str(bbox_shift)])

    print(f"Running MuseTalk...")
    print(f"  Video: {video_path}")
    print(f"  Audio: {audio_path}")
    print(f"  Output dir: {output_path.parent}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(MUSETALK_DIR),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
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
