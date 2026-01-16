"""
Unit tests for audio helper functions.
"""

import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from utils.audio_helpers import read_wav_file, resample_audio, write_wav_file


class TestReadWavFile:
    """Tests for read_wav_file function."""

    def test_read_16bit_mono_wav(self, tmp_path: Path) -> None:
        """Test reading a valid 16-bit mono WAV file."""
        wav_path = tmp_path / "test_16bit_mono.wav"
        sample_rate = 44100
        duration_seconds = 0.1
        n_frames = int(sample_rate * duration_seconds)

        # Create a 16-bit mono WAV file with a sine wave
        frequency = 440  # A4 note
        t = np.linspace(0, duration_seconds, n_frames, dtype=np.float32)
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        # Read the file back
        read_sr, read_audio = read_wav_file(wav_path)

        assert read_sr == sample_rate
        assert len(read_audio) == n_frames
        assert read_audio.dtype == np.float32
        # Values should be normalized to [-1.0, 1.0]
        assert np.max(np.abs(read_audio)) <= 1.0
        # Should roughly match original (within quantization error)
        np.testing.assert_allclose(read_audio, audio_data, atol=1e-4)

    def test_read_stereo_wav_converts_to_mono(self, tmp_path: Path) -> None:
        """Test reading a stereo WAV file converts to mono correctly."""
        wav_path = tmp_path / "test_stereo.wav"
        sample_rate = 22050
        duration_seconds = 0.05
        n_frames = int(sample_rate * duration_seconds)

        # Create stereo audio: left channel = sine, right channel = sine with phase shift
        frequency = 440
        t = np.linspace(0, duration_seconds, n_frames, dtype=np.float32)
        left_channel = np.sin(2 * np.pi * frequency * t)
        right_channel = np.sin(2 * np.pi * frequency * t + np.pi / 4)  # 45 degree phase shift

        # Interleave channels for stereo
        stereo_audio = np.empty(n_frames * 2, dtype=np.float32)
        stereo_audio[0::2] = left_channel
        stereo_audio[1::2] = right_channel
        stereo_int16 = (stereo_audio * 32767).astype(np.int16)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(stereo_int16.tobytes())

        # Read the file back
        read_sr, read_audio = read_wav_file(wav_path)

        assert read_sr == sample_rate
        # Should be mono now
        assert len(read_audio) == n_frames
        # Mono should be average of channels
        expected_mono = (left_channel + right_channel) / 2
        np.testing.assert_allclose(read_audio, expected_mono, atol=1e-4)

    def test_read_32bit_wav(self, tmp_path: Path) -> None:
        """Test reading a 32-bit WAV file."""
        wav_path = tmp_path / "test_32bit.wav"
        sample_rate = 48000
        duration_seconds = 0.05
        n_frames = int(sample_rate * duration_seconds)

        # Create a 32-bit WAV file
        frequency = 880
        t = np.linspace(0, duration_seconds, n_frames, dtype=np.float32)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.5  # Half amplitude
        audio_int32 = (audio_data * 2147483647).astype(np.int32)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(4)  # 32-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int32.tobytes())

        # Read the file back
        read_sr, read_audio = read_wav_file(wav_path)

        assert read_sr == sample_rate
        assert len(read_audio) == n_frames
        assert read_audio.dtype == np.float32
        np.testing.assert_allclose(read_audio, audio_data, atol=1e-6)


class TestWriteWavFile:
    """Tests for write_wav_file function."""

    def test_write_creates_valid_wav(self, tmp_path: Path) -> None:
        """Test that write_wav_file creates a valid WAV file."""
        wav_path = tmp_path / "output.wav"
        sample_rate = 24000
        duration_seconds = 0.1
        n_frames = int(sample_rate * duration_seconds)

        # Create test audio data
        frequency = 440
        t = np.linspace(0, duration_seconds, n_frames, dtype=np.float32)
        audio_data = np.sin(2 * np.pi * frequency * t)

        # Write the file
        write_wav_file(wav_path, sample_rate, audio_data)

        # Verify the file was created
        assert wav_path.exists()

        # Read it back with wave module to verify format
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1  # Mono
            assert wf.getsampwidth() == 2  # 16-bit
            assert wf.getframerate() == sample_rate
            assert wf.getnframes() == n_frames

    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        """Test that write then read produces the same audio."""
        wav_path = tmp_path / "roundtrip.wav"
        sample_rate = 16000
        duration_seconds = 0.2
        n_frames = int(sample_rate * duration_seconds)

        # Create test audio data
        t = np.linspace(0, duration_seconds, n_frames, dtype=np.float32)
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.8

        # Write then read
        write_wav_file(wav_path, sample_rate, audio_data)
        read_sr, read_audio = read_wav_file(wav_path)

        assert read_sr == sample_rate
        assert len(read_audio) == n_frames
        # Should be close (within 16-bit quantization error)
        np.testing.assert_allclose(read_audio, audio_data, atol=1e-4)

    def test_write_clips_values_outside_range(self, tmp_path: Path) -> None:
        """Test that values outside [-1, 1] are clipped correctly."""
        wav_path = tmp_path / "clipped.wav"
        sample_rate = 8000

        # Create audio with values outside valid range
        audio_data = np.array([2.0, -2.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)

        # Write the file
        write_wav_file(wav_path, sample_rate, audio_data)

        # Read it back
        _, read_audio = read_wav_file(wav_path)

        # Values should be clipped to [-1, 1]
        expected = np.array([1.0, -1.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        np.testing.assert_allclose(read_audio, expected, atol=1e-4)


class TestResampleAudio:
    """Tests for resample_audio function."""

    def test_resample_same_rate_returns_unchanged(self) -> None:
        """Test that resampling to the same rate returns unchanged audio."""
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        result = resample_audio(audio, 44100, 44100)

        np.testing.assert_array_equal(result, audio)

    def test_resample_upsample(self) -> None:
        """Test upsampling produces correct length."""
        orig_sr = 22050
        target_sr = 44100
        duration_seconds = 0.1
        n_frames = int(orig_sr * duration_seconds)

        # Create test audio
        audio = np.sin(np.linspace(0, 2 * np.pi, n_frames)).astype(np.float32)

        result = resample_audio(audio, orig_sr, target_sr)

        # Should have double the samples
        expected_length = int(n_frames * target_sr / orig_sr)
        assert len(result) == expected_length
        # Values should still be in valid range
        assert np.max(np.abs(result)) <= 1.0

    def test_resample_downsample(self) -> None:
        """Test downsampling produces correct length."""
        orig_sr = 48000
        target_sr = 16000
        duration_seconds = 0.1
        n_frames = int(orig_sr * duration_seconds)

        # Create test audio
        audio = np.sin(np.linspace(0, 2 * np.pi * 4, n_frames)).astype(np.float32)

        result = resample_audio(audio, orig_sr, target_sr)

        # Should have 1/3 the samples
        expected_length = int(n_frames * target_sr / orig_sr)
        assert len(result) == expected_length

    def test_resample_preserves_dc_level(self) -> None:
        """Test that resampling preserves DC (constant) signal."""
        audio = np.full(1000, 0.5, dtype=np.float32)

        result = resample_audio(audio, 44100, 22050)

        # All values should still be 0.5
        np.testing.assert_allclose(result, 0.5, atol=1e-6)

    def test_resample_to_common_rate(self) -> None:
        """Test resampling to Chatterbox sample rate (24000 Hz)."""
        orig_sr = 44100
        target_sr = 24000
        duration_seconds = 0.5
        n_frames = int(orig_sr * duration_seconds)

        # Create a 440 Hz sine wave
        t = np.linspace(0, duration_seconds, n_frames, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)

        result = resample_audio(audio, orig_sr, target_sr)

        # Check length
        expected_length = int(n_frames * target_sr / orig_sr)
        assert len(result) == expected_length

        # The resampled signal should still oscillate around zero
        assert np.abs(np.mean(result)) < 0.01
