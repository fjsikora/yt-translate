"""
Audio helper functions for WAV file I/O and resampling.
"""

import struct
import wave
from pathlib import Path

import numpy as np


def read_wav_file(wav_path: Path) -> tuple[int, np.ndarray]:
    """
    Read a WAV file and return sample rate and float32 audio data.

    Args:
        wav_path: Path to the WAV file to read

    Returns:
        Tuple of (sample_rate, audio_data) where audio_data is a 1D numpy array
        of float32 values normalized to [-1.0, 1.0]
    """
    with wave.open(str(wav_path), 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        raw_data = wf.readframes(n_frames)

        # Parse based on sample width
        if sample_width == 2:  # 16-bit
            fmt = f'<{n_frames * n_channels}h'
            samples = struct.unpack(fmt, raw_data)
            audio_np = np.array(samples, dtype=np.float32) / 32767.0
        elif sample_width == 4:  # 32-bit
            fmt = f'<{n_frames * n_channels}i'
            samples = struct.unpack(fmt, raw_data)
            audio_np = np.array(samples, dtype=np.float32) / 2147483647.0
        else:
            # Default to 16-bit assumption
            fmt = f'<{len(raw_data) // 2}h'
            samples = struct.unpack(fmt, raw_data)
            audio_np = np.array(samples, dtype=np.float32) / 32767.0

        # Handle stereo -> mono
        if n_channels > 1:
            audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)

        return sample_rate, audio_np


def write_wav_file(wav_path: Path, sample_rate: int, audio: np.ndarray) -> None:
    """
    Write float32 audio data to a WAV file as 16-bit PCM.

    Args:
        wav_path: Path where the WAV file should be written
        sample_rate: Sample rate in Hz
        audio: 1D numpy array of float32 audio samples in [-1.0, 1.0] range
    """
    # Convert to 16-bit integer
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio using linear interpolation.

    Args:
        audio: Input audio as 1D numpy array
        orig_sr: Original sample rate in Hz
        target_sr: Target sample rate in Hz

    Returns:
        Resampled audio as 1D numpy array
    """
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio)
