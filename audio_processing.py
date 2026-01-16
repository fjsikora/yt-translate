#!/usr/bin/env python3
"""
Audio Processing Module for YouTube Video Translator

Provides audio separation and speaker diarization functionality:
- AudioSeparator: Separates vocals from background audio using Demucs
- SpeakerDiarizer: Identifies different speakers using pyannote.audio

Both classes are designed to be used by translate.py and cloud_translate.py.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class AudioSeparator:
    """
    Separates audio into vocals and background (instrumental) tracks using Demucs.

    Uses the htdemucs model which provides high-quality source separation.
    Works on both GPU and CPU.
    """

    def __init__(self, device: Optional[str] = None, model_name: str = "htdemucs"):
        """
        Initialize the audio separator.

        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            model_name: Demucs model name (default: 'htdemucs')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self._model = None
        self._sample_rate: int = 44100  # Demucs default sample rate

    def _load_model(self):
        """Lazily load the Demucs model."""
        if self._model is None:
            try:
                from demucs.pretrained import get_model
                from demucs.apply import apply_model

                self._model = get_model(self.model_name)
                self._model.to(self.device)
                self._model.eval()
            except ImportError as e:
                raise ImportError(
                    "Demucs is not installed. Install with: pip install demucs>=4.0.0"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load Demucs model '{self.model_name}': {e}") from e

    def separate(
        self,
        audio_path: Path,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Separate audio into vocals and background tracks.

        Args:
            audio_path: Path to the input audio file

        Returns:
            Tuple of (vocals, background, sample_rate) where:
            - vocals: numpy array of vocal audio
            - background: numpy array of background/instrumental audio
            - sample_rate: sample rate of the output audio

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If separation fails
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            self._load_model()

            from demucs.audio import AudioFile
            from demucs.apply import apply_model

            # Load audio
            audio_file = AudioFile(audio_path)
            wav = audio_file.read(
                seek_time=0,
                duration=-1,  # Full duration
                streams=0,
            )

            # Ensure correct shape and device
            # wav shape should be (channels, samples)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if wav.shape[0] == 1:
                # Mono to stereo
                wav = wav.repeat(2, 1)

            # Add batch dimension
            wav = wav.unsqueeze(0).to(self.device)

            # Get sample rate from model
            self._sample_rate = self._model.samplerate

            # Apply model
            with torch.no_grad():
                sources = apply_model(
                    self._model,
                    wav,
                    device=self.device,
                    progress=False,
                )

            # sources shape: (batch, sources, channels, samples)
            # htdemucs sources: drums, bass, other, vocals
            sources = sources.squeeze(0)  # Remove batch dimension

            # Get source names
            source_names = self._model.sources
            vocals_idx = source_names.index("vocals") if "vocals" in source_names else -1

            if vocals_idx == -1:
                raise RuntimeError("Model doesn't have 'vocals' source")

            # Extract vocals and combine others for background
            vocals = sources[vocals_idx].cpu().numpy()

            # Background = sum of all non-vocal sources
            background_sources = [
                sources[i] for i in range(len(source_names)) if i != vocals_idx
            ]
            background = sum(background_sources).cpu().numpy()

            # Convert stereo to mono if needed (average channels)
            if vocals.ndim == 2 and vocals.shape[0] == 2:
                vocals = vocals.mean(axis=0)
            if background.ndim == 2 and background.shape[0] == 2:
                background = background.mean(axis=0)

            return vocals, background, self._sample_rate

        except ImportError:
            raise
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Audio separation failed: {e}") from e

    def save_separated(
        self,
        vocals: np.ndarray,
        background: np.ndarray,
        sample_rate: int,
        output_dir: Path,
        vocals_filename: str = "vocals.wav",
        background_filename: str = "background.wav",
    ) -> tuple[Path, Path]:
        """
        Save separated audio tracks to files.

        Args:
            vocals: Numpy array of vocal audio
            background: Numpy array of background audio
            sample_rate: Sample rate of the audio
            output_dir: Directory to save the files
            vocals_filename: Filename for vocals (default: 'vocals.wav')
            background_filename: Filename for background (default: 'background.wav')

        Returns:
            Tuple of (vocals_path, background_path)
        """
        import scipy.io.wavfile as wavfile

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vocals_path = output_dir / vocals_filename
        background_path = output_dir / background_filename

        # Normalize and convert to int16
        def normalize_and_convert(audio: np.ndarray) -> np.ndarray:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95  # Leave headroom
            return (audio * 32767).astype(np.int16)

        vocals_int16 = normalize_and_convert(vocals)
        background_int16 = normalize_and_convert(background)

        wavfile.write(str(vocals_path), sample_rate, vocals_int16)
        wavfile.write(str(background_path), sample_rate, background_int16)

        return vocals_path, background_path


class SpeakerDiarizer:
    """
    Identifies different speakers in audio using pyannote.audio.

    Uses the pyannote/speaker-diarization-3.1 model for speaker identification.
    Requires a HuggingFace token with access to the pyannote models.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the speaker diarizer.

        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            hf_token: HuggingFace token for accessing pyannote models.
                      Falls back to HUGGINGFACE_TOKEN environment variable.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
        self._pipeline = None
        self._diarization = None

    def _load_pipeline(self):
        """Lazily load the diarization pipeline."""
        if self._pipeline is None:
            if not self.hf_token:
                raise ValueError(
                    "HuggingFace token required for pyannote models. "
                    "Set HUGGINGFACE_TOKEN environment variable or pass hf_token parameter."
                )

            try:
                from pyannote.audio import Pipeline

                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token,
                )
                self._pipeline.to(torch.device(self.device))
            except ImportError as e:
                raise ImportError(
                    "pyannote.audio is not installed. Install with: pip install pyannote.audio>=3.1.0"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load diarization pipeline: {e}") from e

    def diarize(self, audio_path: Path) -> list[dict]:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of segment dicts with keys:
            - 'speaker': Speaker label (e.g., 'SPEAKER_00', 'SPEAKER_01')
            - 'start': Segment start time in seconds
            - 'end': Segment end time in seconds
            - 'duration': Segment duration in seconds

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If diarization fails
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            self._load_pipeline()

            # Run diarization
            self._diarization = self._pipeline(str(audio_path))

            # Convert to list of segments
            segments = []
            for turn, _, speaker in self._diarization.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.end - turn.start,
                })

            # Sort by start time
            segments.sort(key=lambda x: x["start"])

            return segments

        except ImportError:
            raise
        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {e}") from e

    def get_speaker_segments(self, segments: Optional[list[dict]] = None) -> dict[str, list[dict]]:
        """
        Group segments by speaker.

        Args:
            segments: List of diarization segments. If None, uses last diarization result.

        Returns:
            Dict mapping speaker ID to list of their segments.
            Example: {'SPEAKER_00': [{'start': 0.0, 'end': 5.0, ...}, ...]}

        Raises:
            ValueError: If no segments available
        """
        if segments is None:
            if self._diarization is None:
                raise ValueError("No diarization results. Run diarize() first.")
            # Re-extract from stored diarization
            segments = []
            for turn, _, speaker in self._diarization.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.end - turn.start,
                })

        speaker_segments: dict[str, list[dict]] = {}
        for seg in segments:
            speaker = seg["speaker"]
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(seg)

        return speaker_segments

    def extract_speaker_samples(
        self,
        audio_path: Path,
        output_dir: Path,
        segments: Optional[list[dict]] = None,
        target_duration: float = 12.0,
        min_duration: float = 3.0,
        target_sample_rate: int = 24000,
    ) -> dict[str, Path]:
        """
        Extract voice samples for each speaker for voice cloning.

        Finds the longest contiguous segment for each speaker and extracts
        a sample suitable for Chatterbox voice cloning (24kHz mono WAV).

        Args:
            audio_path: Path to the source audio file
            output_dir: Directory to save voice samples
            segments: Diarization segments. If None, uses last diarization result.
            target_duration: Target sample duration in seconds (default: 12.0)
            min_duration: Minimum acceptable duration (default: 3.0)
            target_sample_rate: Output sample rate (default: 24000 for Chatterbox)

        Returns:
            Dict mapping speaker ID to voice sample file path
            Example: {'SPEAKER_00': Path('output/speaker_00_sample.wav'), ...}

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If extraction fails
        """
        import subprocess

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get speaker segments
        speaker_segs = self.get_speaker_segments(segments)

        speaker_samples: dict[str, Path] = {}

        for speaker, segs in speaker_segs.items():
            # Find the longest segment for this speaker
            longest_seg = max(segs, key=lambda x: x["duration"])

            # Calculate duration to extract
            available_duration = longest_seg["duration"]
            extract_duration = min(target_duration, available_duration)

            # Skip if segment is too short
            if extract_duration < min_duration:
                continue

            # Clamp to 10-15 second range if we have enough
            if extract_duration > 15.0:
                extract_duration = 15.0

            # Output path
            safe_speaker = speaker.replace(" ", "_").lower()
            sample_path = output_dir / f"{safe_speaker}_sample.wav"

            try:
                # Use ffmpeg to extract segment at 24kHz mono
                cmd = [
                    "ffmpeg",
                    "-i", str(audio_path),
                    "-ss", str(longest_seg["start"]),
                    "-t", str(extract_duration),
                    "-ar", str(target_sample_rate),
                    "-ac", "1",
                    "-y",
                    str(sample_path),
                ]
                subprocess.run(cmd, capture_output=True, check=True)

                speaker_samples[speaker] = sample_path

            except subprocess.CalledProcessError as e:
                # Log but continue with other speakers
                print(f"Warning: Failed to extract sample for {speaker}: {e}")
                continue

        return speaker_samples
