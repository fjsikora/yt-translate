#!/usr/bin/env python3
"""
YouTube Video Translator

Translates YouTube videos to other languages using:
- yt-dlp: Download video & extract audio
- Whisper: Speech-to-text transcription
- deep-translator: Text translation (Google backend)
- Chatterbox: Voice cloning TTS
- ffmpeg: Audio/video merging
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Callable

# Force GPU 1 (RTX 5060 Ti) before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import whisper
import yt_dlp
from deep_translator import GoogleTranslator
from audio_processing import AudioSeparator, SpeakerDiarizer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from languages import (
    SUPPORTED_LANGUAGES_BY_NAME as SUPPORTED_LANGUAGES,
    GOOGLE_LANG_CODES_BY_NAME as GOOGLE_LANG_CODES,
    ISO_639_1_TO_639_2,
)

WHISPER_MODEL = "base"
OUTPUT_DIR = Path("output")
console = Console()


class ProgressTracker:
    """Track and display progress for the translation pipeline."""

    STAGES = [
        ("download", "Download Video"),
        ("separate", "Separate Audio"),
        ("diarize", "Identify Speakers"),
        ("transcribe", "Transcribe Audio"),
        ("translate", "Translate Text"),
        ("synthesize", "Synthesize Voice"),
        ("mix", "Mix Audio"),
        ("merge", "Merge Audio/Video"),
    ]

    def __init__(self, video_url: str, target_lang: str):
        self.video_url = video_url
        self.target_lang = target_lang
        self.current_stage = 0
        self.stage_status = {s[0]: "pending" for s in self.STAGES}
        self.stage_details = {s[0]: "" for s in self.STAGES}
        self.stage_times = {s[0]: 0.0 for s in self.STAGES}
        self.stage_start_time = None
        self.start_time = time.time()

    def start_stage(self, stage_id: str, detail: str = ""):
        """Mark a stage as in progress."""
        self.stage_status[stage_id] = "running"
        self.stage_details[stage_id] = detail
        self.stage_start_time = time.time()

    def update_detail(self, stage_id: str, detail: str):
        """Update the detail text for a stage."""
        self.stage_details[stage_id] = detail

    def complete_stage(self, stage_id: str, detail: str = ""):
        """Mark a stage as complete."""
        if self.stage_start_time:
            self.stage_times[stage_id] = time.time() - self.stage_start_time
        self.stage_status[stage_id] = "done"
        self.stage_details[stage_id] = detail
        self.current_stage += 1

    def fail_stage(self, stage_id: str, error: str):
        """Mark a stage as failed."""
        self.stage_status[stage_id] = "failed"
        self.stage_details[stage_id] = f"Error: {error}"

    def _format_time(self, seconds: float) -> str:
        """Format seconds as mm:ss or hh:mm:ss."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins, secs = divmod(int(seconds), 60)
            return f"{mins}m {secs}s"
        else:
            hours, remainder = divmod(int(seconds), 3600)
            mins, secs = divmod(remainder, 60)
            return f"{hours}h {mins}m {secs}s"

    def render(self) -> Panel:
        """Render the progress display as a Rich Panel."""
        lines = []

        # Header info
        lines.append(Text(f"  Video: {self.video_url[:50]}...", style="dim"))
        lines.append(Text(f"  Target: {self.target_lang}", style="dim"))
        lines.append(Text(""))

        # Progress bar
        completed = sum(1 for s in self.stage_status.values() if s == "done")
        total = len(self.STAGES)
        pct = int((completed / total) * 100)
        bar_width = 30
        filled = int((completed / total) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(Text(f"  [{bar}] {pct}% ({completed}/{total})", style="bold cyan"))
        lines.append(Text(""))

        # Stage list
        for stage_id, stage_name in self.STAGES:
            status = self.stage_status[stage_id]
            detail = self.stage_details[stage_id]
            elapsed = self.stage_times[stage_id]

            if status == "done":
                icon = "✓"
                icon_style = "green"
                name_style = "green"
                time_str = f" ({self._format_time(elapsed)})"
            elif status == "running":
                icon = "●"
                icon_style = "cyan bold"
                name_style = "cyan bold"
                if self.stage_start_time:
                    running_time = time.time() - self.stage_start_time
                    time_str = f" ({self._format_time(running_time)})"
                else:
                    time_str = ""
            elif status == "failed":
                icon = "✗"
                icon_style = "red"
                name_style = "red"
                time_str = ""
            else:
                icon = "○"
                icon_style = "dim"
                name_style = "dim"
                time_str = ""

            line = Text()
            line.append("  ", style="")
            line.append(icon, style=icon_style)
            line.append(" ", style="")
            line.append(stage_name, style=name_style)
            if time_str:
                line.append(time_str, style="dim")

            if detail and status == "running":
                lines.append(line)
                lines.append(Text(f"      {detail}", style="dim italic"))
            elif detail and status == "done":
                lines.append(line)
                if detail and not detail.startswith("Error"):
                    lines.append(Text(f"      {detail[:60]}{'...' if len(detail) > 60 else ''}", style="dim"))
            else:
                lines.append(line)

        # Total elapsed time
        total_elapsed = time.time() - self.start_time
        lines.append(Text(""))
        lines.append(Text(f"  Total time: {self._format_time(total_elapsed)}", style="dim"))

        return Panel(
            Group(*lines),
            title="[bold blue]YouTube Video Translator[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def download_youtube(url: str, tracker: ProgressTracker, update_fn: Callable) -> tuple[Path, Path, str]:
    """Download YouTube video and extract audio."""
    ensure_output_dir()

    tracker.update_detail("download", "Fetching video info...")
    update_fn()

    # First, get video info to extract ID
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info['id']
        title = info.get('title', video_id)
        duration = info.get('duration', 0)
        duration_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration else "unknown"

    tracker.update_detail("download", f"Found: {title[:35]}... ({duration_str})")
    update_fn()

    video_path = OUTPUT_DIR / f"{video_id}.mp4"
    audio_path = OUTPUT_DIR / f"{video_id}.wav"

    # Check if video already exists
    existing_video = None
    possible_paths = [
        OUTPUT_DIR / f"{video_id}.mp4",
        OUTPUT_DIR / f"{video_id}.mkv",
        OUTPUT_DIR / f"{video_id}.webm",
    ]
    for p in possible_paths:
        if p.exists():
            existing_video = p
            break

    if existing_video:
        tracker.update_detail("download", f"Using cached: {title[:35]}...")
        update_fn()
        video_path = existing_video
    else:
        tracker.update_detail("download", f"Downloading: {title[:35]}...")
        update_fn()

        # Download video with progress hook
        def progress_hook(d):
            if d['status'] == 'downloading':
                pct = d.get('_percent_str', '?%').strip()
                speed = d.get('_speed_str', '').strip()
                tracker.update_detail("download", f"Downloading: {pct} @ {speed}")
                update_fn()

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(OUTPUT_DIR / f"{video_id}.%(ext)s"),
            'quiet': True,
            'no_warnings': True,
            'merge_output_format': 'mp4',
            'progress_hooks': [progress_hook],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the downloaded video file
        for p in possible_paths:
            if p.exists():
                video_path = p
                break

    # Check if audio already exists
    if audio_path.exists():
        tracker.update_detail("download", "Using cached audio...")
        update_fn()
    else:
        tracker.update_detail("download", "Extracting audio with ffmpeg...")
        update_fn()

        # Extract audio using ffmpeg
        subprocess.run([
            'ffmpeg', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-y', str(audio_path)
        ], capture_output=True, check=True)

    return video_path, audio_path, title


def transcribe_audio(audio_path: Path, tracker: ProgressTracker, update_fn: Callable) -> list[dict]:
    """Transcribe audio using local Whisper model. Returns segments with timestamps."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = "GPU" if device == "cuda" else "CPU"

    # Get audio duration
    import wave
    try:
        with wave.open(str(audio_path), 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration_secs = frames / float(rate)
            duration_str = f"{int(duration_secs // 60)}m {int(duration_secs % 60)}s"
    except Exception:
        duration_secs = 0
        duration_str = "unknown"

    tracker.update_detail("transcribe", f"Loading Whisper model ({WHISPER_MODEL}) on {device_name}...")
    update_fn()

    model = whisper.load_model(WHISPER_MODEL, device=device)

    tracker.update_detail("transcribe", f"Transcribing {duration_str} of audio on {device_name}...")
    update_fn()

    # Get segments with timestamps
    result = model.transcribe(str(audio_path))

    # Return segments with timing info
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "duration": seg["end"] - seg["start"]
        })

    return segments


def translate_segments(segments: list[dict], target_lang: str, tracker: ProgressTracker, update_fn: Callable) -> list[dict]:
    """Translate each segment while preserving timing info."""
    lang_code = GOOGLE_LANG_CODES.get(target_lang, target_lang.lower()[:2])

    translated_segments = []
    total = len(segments)

    for i, seg in enumerate(segments):
        tracker.update_detail("translate", f"Translating segment {i+1}/{total}...")
        update_fn()

        text = seg["text"]
        if text:
            try:
                translated_text = GoogleTranslator(source='auto', target=lang_code).translate(text)
            except Exception:
                translated_text = text  # Fallback to original if translation fails
        else:
            translated_text = ""

        translated_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["duration"],
            "original_text": text,
            "translated_text": translated_text
        })

    return translated_segments


def synthesize_segments(segments: list[dict], voice_sample: Path, lang_code: str,
                        total_duration: float, tracker: ProgressTracker, update_fn: Callable) -> Path:
    """Generate speech for each segment at natural speed without time-stretching.

    Audio plays at natural 1x speed. Segment start times are adjusted to prevent
    overlap - if a segment hasn't finished, the next one waits for it to complete.
    This eliminates echo/reverb artifacts caused by time-stretching.
    """
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = "GPU" if device == "cuda" else "CPU"

    tracker.update_detail("synthesize", f"Loading Chatterbox multilingual model on {device_name}...")
    update_fn()

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    sample_rate = model.sr

    tracker.update_detail("synthesize", f"Analyzing voice from {voice_sample.name}...")
    update_fn()

    # Process each segment and collect audio with original start times
    total_segments = len(segments)
    segment_audios: list[tuple[float, np.ndarray]] = []  # (original_start, audio)

    for i, seg in enumerate(segments):
        text = seg["translated_text"]

        if not text:
            # Skip empty segments entirely
            continue

        tracker.update_detail("synthesize", f"Generating segment {i+1}/{total_segments}: {text[:30]}...")
        update_fn()

        # Generate speech for this segment at natural speed (no time-stretching)
        wav = model.generate(
            text,
            audio_prompt_path=str(voice_sample),
            language_id=lang_code,
        )

        # Convert to numpy
        audio_np = wav.squeeze().cpu().numpy()
        segment_audios.append((seg["start"], audio_np))

    # Combine all segments with adjusted timing to prevent overlap
    # Each segment starts at max(original_start, previous_segment_end)
    tracker.update_detail("synthesize", "Combining segments (natural timing, no stretching)...")
    update_fn()

    # First pass: calculate actual positions to determine total length
    positioned_audios: list[tuple[float, np.ndarray]] = []
    current_end_time = 0.0

    for original_start, audio in segment_audios:
        audio_duration = len(audio) / sample_rate

        # Start at the later of: original timestamp or when previous segment ends
        actual_start = max(original_start, current_end_time)
        positioned_audios.append((actual_start, audio))

        # Update end time for next segment
        current_end_time = actual_start + audio_duration

    # Calculate actual output length (may be longer than original video)
    actual_total_duration = max(total_duration, current_end_time)
    output_length = int(actual_total_duration * sample_rate)
    output_audio = np.zeros(output_length)

    # Second pass: place audio at calculated positions
    for start_time, audio in positioned_audios:
        start_sample = int(start_time * sample_rate)
        end_sample = start_sample + len(audio)

        # Ensure we don't overflow (shouldn't happen with calculated length)
        if end_sample > output_length:
            audio = audio[:output_length - start_sample]

        # Place audio (no overlap possible with our timing logic)
        output_audio[start_sample:start_sample + len(audio)] = audio

    # Normalize to prevent clipping
    max_val = np.max(np.abs(output_audio))
    if max_val > 0.99:
        output_audio = output_audio * 0.99 / max_val

    # Save the combined audio
    output_path = OUTPUT_DIR / "translated_audio.wav"
    import scipy.io.wavfile as wav
    audio_int16 = (output_audio * 32767).astype(np.int16)
    wav.write(str(output_path), sample_rate, audio_int16)

    audio_duration = len(output_audio) / sample_rate
    tracker.update_detail("synthesize", f"Saved {audio_duration:.1f}s audio (natural speed)")
    update_fn()

    return output_path


def synthesize_segments_multi_speaker(
    segments: list[dict],
    diarization_segments: list[dict],
    speaker_samples: dict[str, Path],
    lang_code: str,
    total_duration: float,
    tracker: ProgressTracker,
    update_fn: Callable,
) -> Path:
    """Generate speech for each segment using speaker-matched voice samples.

    Maps each transcription segment to its speaker via diarization timestamps,
    then uses the appropriate voice sample for Chatterbox generation.

    Audio plays at natural 1x speed. Segment start times are adjusted to prevent
    overlap - if a segment hasn't finished, the next one waits for it to complete.

    Args:
        segments: Translated text segments with start/end times
        diarization_segments: Speaker diarization results with speaker labels
        speaker_samples: Dict mapping speaker ID to voice sample path
        lang_code: Target language code for Chatterbox
        total_duration: Original audio duration in seconds
        tracker: Progress tracker for UI updates
        update_fn: Callback to update the UI

    Returns:
        Path to the generated audio file
    """
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = "GPU" if device == "cuda" else "CPU"

    tracker.update_detail("synthesize", f"Loading Chatterbox multilingual model on {device_name}...")
    update_fn()

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    sample_rate = model.sr

    # Get fallback sample (first available)
    if not speaker_samples:
        raise ValueError("No speaker samples available for synthesis")

    fallback_speaker = next(iter(speaker_samples))
    fallback_sample = speaker_samples[fallback_speaker]

    tracker.update_detail("synthesize", f"Found {len(speaker_samples)} speaker(s), matching to segments...")
    update_fn()

    def match_segment_to_speaker(seg_start: float, seg_end: float) -> str:
        """Find the speaker with maximum overlap for a given segment time range."""
        best_speaker = fallback_speaker
        best_overlap = 0.0

        for diar_seg in diarization_segments:
            # Calculate overlap between transcription segment and diarization segment
            overlap_start = max(seg_start, diar_seg["start"])
            overlap_end = min(seg_end, diar_seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg["speaker"]

        return best_speaker

    # Process each segment and collect audio with original start times
    total_segments = len(segments)
    segment_audios: list[tuple[float, np.ndarray]] = []  # (original_start, audio)

    for i, seg in enumerate(segments):
        text = seg["translated_text"]

        if not text:
            # Skip empty segments entirely
            continue

        # Match this segment to a speaker
        speaker = match_segment_to_speaker(seg["start"], seg["end"])

        # Get voice sample for this speaker (with fallback)
        voice_sample = speaker_samples.get(speaker, fallback_sample)

        tracker.update_detail(
            "synthesize",
            f"Segment {i+1}/{total_segments} ({speaker}): {text[:25]}..."
        )
        update_fn()

        # Generate speech for this segment at natural speed (no time-stretching)
        wav = model.generate(
            text,
            audio_prompt_path=str(voice_sample),
            language_id=lang_code,
        )

        # Convert to numpy
        audio_np = wav.squeeze().cpu().numpy()
        segment_audios.append((seg["start"], audio_np))

    # Combine all segments with adjusted timing to prevent overlap
    # Each segment starts at max(original_start, previous_segment_end)
    tracker.update_detail("synthesize", "Combining segments (natural timing, no stretching)...")
    update_fn()

    # First pass: calculate actual positions to determine total length
    positioned_audios: list[tuple[float, np.ndarray]] = []
    current_end_time = 0.0

    for original_start, audio in segment_audios:
        audio_duration = len(audio) / sample_rate

        # Start at the later of: original timestamp or when previous segment ends
        actual_start = max(original_start, current_end_time)
        positioned_audios.append((actual_start, audio))

        # Update end time for next segment
        current_end_time = actual_start + audio_duration

    # Calculate actual output length (may be longer than original video)
    actual_total_duration = max(total_duration, current_end_time)
    output_length = int(actual_total_duration * sample_rate)
    output_audio = np.zeros(output_length)

    # Second pass: place audio at calculated positions
    for start_time, audio in positioned_audios:
        start_sample = int(start_time * sample_rate)
        end_sample = start_sample + len(audio)

        # Ensure we don't overflow (shouldn't happen with calculated length)
        if end_sample > output_length:
            audio = audio[:output_length - start_sample]

        # Place audio (no overlap possible with our timing logic)
        output_audio[start_sample:start_sample + len(audio)] = audio

    # Normalize to prevent clipping
    max_val = np.max(np.abs(output_audio))
    if max_val > 0.99:
        output_audio = output_audio * 0.99 / max_val

    # Save the combined audio
    output_path = OUTPUT_DIR / "translated_audio.wav"
    import scipy.io.wavfile as wav
    audio_int16 = (output_audio * 32767).astype(np.int16)
    wav.write(str(output_path), sample_rate, audio_int16)

    audio_duration = len(output_audio) / sample_rate
    tracker.update_detail("synthesize", f"Saved {audio_duration:.1f}s audio ({len(speaker_samples)} speakers)")
    update_fn()

    return output_path


def generate_srt(segments: list[dict], output_path: Optional[Path] = None) -> Path:
    """Generate SRT subtitle file from translated segments.

    Args:
        segments: List of translated segments with start, end, and translated_text fields
        output_path: Output path for the SRT file. If None, uses OUTPUT_DIR/"subtitles.srt"

    Returns:
        Path to the generated SRT file

    SRT format:
        1
        00:00:00,000 --> 00:00:05,123
        Subtitle text here

        2
        00:00:05,500 --> 00:00:10,000
        Next subtitle text
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "subtitles.srt"

    def format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    srt_lines: list[str] = []

    subtitle_index = 1
    for seg in segments:
        text = seg.get("translated_text", "").strip()

        # Skip empty segments
        if not text:
            continue

        start_time = seg.get("start", 0.0)
        end_time = seg.get("end", 0.0)

        # Add SRT entry
        srt_lines.append(str(subtitle_index))
        srt_lines.append(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}")
        srt_lines.append(text)
        srt_lines.append("")  # Blank line between entries

        subtitle_index += 1

    # Write with UTF-8 encoding for Unicode support
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    return output_path


def mix_audio_with_background(
    speech_path: Path,
    background_path: Path,
    output_path: Optional[Path] = None,
    background_volume: float = 0.3,
) -> Path:
    """Mix translated speech with original background audio using ffmpeg.

    Args:
        speech_path: Path to the translated speech WAV file (100% volume)
        background_path: Path to the background audio WAV file
        output_path: Output path for the mixed audio. If None, uses OUTPUT_DIR/"mixed_audio.wav"
        background_volume: Volume level for background audio (0.0 to 1.0, default 0.3 = 30%)

    Returns:
        Path to the mixed audio file

    The output duration matches the speech track. If background is shorter,
    it's padded with silence. If background is longer, it's trimmed.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "mixed_audio.wav"

    # Clamp background volume to valid range
    background_volume = max(0.0, min(1.0, background_volume))

    # Use ffmpeg to mix the audio tracks
    # - First input (speech) at full volume
    # - Second input (background) at configurable volume
    # - Output duration matches the speech track (shortest=0 with apad on background)
    subprocess.run([
        'ffmpeg',
        '-i', str(speech_path),
        '-i', str(background_path),
        '-filter_complex',
        f'[1:a]apad,volume={background_volume}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0',
        '-ac', '2',  # Stereo output
        '-y',
        str(output_path)
    ], capture_output=True, check=True)

    return output_path


def get_iso_639_2_code(lang_code: str) -> str:
    """Convert ISO 639-1 (2-letter) language code to ISO 639-2 (3-letter) code.

    Args:
        lang_code: 2-letter or 3-letter language code

    Returns:
        3-letter ISO 639-2 language code for ffmpeg metadata
    """
    if len(lang_code) == 3:
        return lang_code
    return ISO_639_1_TO_639_2.get(lang_code, "eng")


def merge_audio_video(video_path: Path, audio_path: Path, output_name: str,
                      tracker: ProgressTracker, update_fn: Callable,
                      subtitle_path: Optional[Path] = None,
                      subtitle_lang: str = "eng") -> Path:
    """Merge translated audio with original video, optionally embedding subtitles.

    Args:
        video_path: Path to the video file
        audio_path: Path to the translated audio file
        output_name: Base name for the output file
        tracker: Progress tracker for UI updates
        update_fn: Callback function for progress updates
        subtitle_path: Optional path to SRT subtitle file
        subtitle_lang: ISO 639-2 language code for subtitle metadata (default: "eng")

    Returns:
        Path to the merged output video file
    """
    output_path = OUTPUT_DIR / f"{output_name}_translated.mp4"

    # Get file sizes for progress info
    def format_size(size_bytes):
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    video_size = format_size(video_path.stat().st_size)
    audio_size = format_size(audio_path.stat().st_size)

    if subtitle_path and subtitle_path.exists():
        tracker.update_detail("merge", f"Merging video ({video_size}) + audio ({audio_size}) + subtitles...")
    else:
        tracker.update_detail("merge", f"Merging video ({video_size}) + audio ({audio_size})...")
    update_fn()

    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-i', str(audio_path),
    ]

    # Add subtitle input if provided
    if subtitle_path and subtitle_path.exists():
        cmd.extend(['-i', str(subtitle_path)])

    # Video codec: copy (no re-encoding)
    cmd.extend(['-c:v', 'copy'])

    # Map video from first input
    cmd.extend(['-map', '0:v:0'])

    # Map audio from second input
    cmd.extend(['-map', '1:a:0'])

    # Map and encode subtitles if provided
    if subtitle_path and subtitle_path.exists():
        # Map subtitle from third input
        cmd.extend(['-map', '2:0'])
        # Use mov_text codec for MP4 soft subs (toggleable)
        cmd.extend(['-c:s', 'mov_text'])
        # Set subtitle language metadata
        cmd.extend(['-metadata:s:s:0', f'language={subtitle_lang}'])

    cmd.extend([
        '-shortest',
        '-y',
        str(output_path)
    ])

    subprocess.run(cmd, capture_output=True, check=True)

    # Show output file size
    output_size = format_size(output_path.stat().st_size)
    if subtitle_path and subtitle_path.exists():
        tracker.update_detail("merge", f"Created {output_path.name} ({output_size}) with subtitles")
    else:
        tracker.update_detail("merge", f"Created {output_path.name} ({output_size})")
    update_fn()

    return output_path


def prompt_url() -> str:
    """Prompt user for YouTube URL."""
    console.print()
    url = Prompt.ask("[bold green]Enter YouTube video URL")
    return url.strip()


def prompt_language() -> tuple[str, str]:
    """Prompt user to select target language."""
    console.print()

    # Display available languages
    table = Table(title="Available Languages", show_header=True)
    table.add_column("Language", style="cyan")
    table.add_column("Code", style="green")

    languages = sorted(SUPPORTED_LANGUAGES.keys())

    # Display in two columns
    mid = (len(languages) + 1) // 2
    for i in range(mid):
        lang1 = languages[i]
        code1 = SUPPORTED_LANGUAGES[lang1]

        if i + mid < len(languages):
            lang2 = languages[i + mid]
            code2 = SUPPORTED_LANGUAGES[lang2]
            table.add_row(f"{lang1} ({code1})", f"{lang2} ({code2})")
        else:
            table.add_row(f"{lang1} ({code1})", "")

    console.print(table)
    console.print()

    # Get user selection
    while True:
        selection = Prompt.ask(
            "[bold green]Enter target language",
            default="Spanish"
        )

        # Check if it's a valid language name or code
        selection_lower = selection.lower()

        for lang_name, lang_code in SUPPORTED_LANGUAGES.items():
            if selection_lower == lang_name.lower() or selection_lower == lang_code:
                return lang_name, lang_code

        console.print(f"[red]Invalid language: {selection}. Please try again.")


def check_dependencies():
    """Check that required system dependencies are available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[red]Error: ffmpeg is not installed or not in PATH")
        console.print("Please install ffmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)


def main():
    """Main entry point."""
    console.print(Panel.fit(
        "[bold blue]YouTube Video Translator[/bold blue]\n"
        "[dim]Powered by Whisper, Chatterbox & Google Translate[/dim]",
        border_style="blue"
    ))

    # Check dependencies
    check_dependencies()

    # Show GPU status
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[green]✓[/green] GPU detected: {gpu_name}")
    else:
        console.print("[yellow]⚠[/yellow] No GPU detected, using CPU (slower)")

    # Get user input
    url = prompt_url()
    lang_name, lang_code = prompt_language()

    console.print()

    # Create progress tracker
    tracker = ProgressTracker(url, lang_name)

    # Run the pipeline with live display
    with Live(tracker.render(), console=console, refresh_per_second=4) as live:
        def update():
            live.update(tracker.render())

        try:
            # Stage 1: Download
            tracker.start_stage("download", "Starting download...")
            update()
            video_path, audio_path, title = download_youtube(url, tracker, update)
            tracker.complete_stage("download", f"Downloaded: {title[:50]}")
            update()

            # Stage 2: Separate audio (vocals from background)
            tracker.start_stage("separate", "Loading Demucs model...")
            update()
            separator = AudioSeparator()
            tracker.update_detail("separate", "Separating vocals from background...")
            update()
            vocals, background, sep_sample_rate = separator.separate(audio_path)
            tracker.update_detail("separate", "Saving separated audio files...")
            update()
            vocals_path, background_path = separator.save_separated(
                vocals, background, sep_sample_rate, OUTPUT_DIR
            )
            tracker.complete_stage("separate", f"Separated: {vocals_path.name}, {background_path.name}")
            update()

            # Stage 3: Speaker diarization (identify speakers in vocals)
            tracker.start_stage("diarize", "Loading speaker diarization model...")
            update()
            diarizer = SpeakerDiarizer()
            tracker.update_detail("diarize", "Identifying speakers in audio...")
            update()
            diarization_segments = diarizer.diarize(vocals_path)
            tracker.update_detail("diarize", "Extracting voice samples per speaker...")
            update()
            speaker_samples = diarizer.extract_speaker_samples(
                vocals_path, OUTPUT_DIR, diarization_segments
            )
            num_speakers = len(speaker_samples)
            tracker.complete_stage("diarize", f"Found {num_speakers} speaker(s)")
            update()

            # Stage 4: Transcribe (using isolated vocals for cleaner input)
            tracker.start_stage("transcribe", "Loading Whisper model...")
            update()
            segments = transcribe_audio(vocals_path, tracker, update)
            total_chars = sum(len(s["text"]) for s in segments)
            tracker.complete_stage("transcribe", f"{len(segments)} segments, {total_chars} chars")
            update()

            # Get total audio duration for synthesis
            if segments:
                total_duration = segments[-1]["end"]
            else:
                total_duration = 0

            # Stage 5: Translate (segment by segment)
            tracker.start_stage("translate", "Starting translation...")
            update()
            translated_segments = translate_segments(segments, lang_name, tracker, update)
            translated_chars = sum(len(s["translated_text"]) for s in translated_segments)
            tracker.complete_stage("translate", f"{len(translated_segments)} segments translated")
            update()

            # Stage 6: Voice synthesis (multi-speaker with per-speaker voice samples)
            tracker.start_stage("synthesize", "Loading TTS model...")
            update()
            if num_speakers > 1 and speaker_samples:
                # Use multi-speaker synthesis
                speech_audio = synthesize_segments_multi_speaker(
                    translated_segments,
                    diarization_segments,
                    speaker_samples,
                    lang_code,
                    total_duration,
                    tracker,
                    update,
                )
            elif speaker_samples:
                # Single speaker - use the one speaker sample
                first_speaker = next(iter(speaker_samples))
                voice_sample = speaker_samples[first_speaker]
                speech_audio = synthesize_segments(
                    translated_segments, voice_sample, lang_code, total_duration, tracker, update
                )
            else:
                # Fallback to original audio as voice sample
                speech_audio = synthesize_segments(
                    translated_segments, audio_path, lang_code, total_duration, tracker, update
                )
            tracker.complete_stage("synthesize", f"Generated {speech_audio.name}")
            update()

            # Stage 7: Mix audio (translated speech + original background)
            tracker.start_stage("mix", "Mixing speech with background audio...")
            update()
            mixed_audio = mix_audio_with_background(
                speech_audio, background_path, background_volume=0.3
            )
            tracker.complete_stage("mix", f"Mixed: {mixed_audio.name}")
            update()

            # Stage 8: Generate subtitles and merge everything
            tracker.start_stage("merge", "Generating subtitles...")
            update()
            subtitle_path = generate_srt(translated_segments)
            tracker.update_detail("merge", "Merging video, audio, and subtitles...")
            update()
            safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:50]
            subtitle_lang_code = get_iso_639_2_code(lang_code)
            output = merge_audio_video(
                video_path, mixed_audio, safe_title, tracker, update,
                subtitle_path=subtitle_path, subtitle_lang=subtitle_lang_code
            )
            tracker.complete_stage("merge", f"Output: {output.name}")
            update()

        except Exception as e:
            # Find the currently running stage and mark it as failed
            for stage_id, status in tracker.stage_status.items():
                if status == "running":
                    tracker.fail_stage(stage_id, str(e))
                    break
            update()
            console.print(f"\n[red]Error: {e}[/red]")
            raise

    # Final success message
    console.print()
    console.print(Panel.fit(
        f"[bold green]Translation Complete![/bold green]\n\n"
        f"[bold]Output:[/bold] {output}\n"
        f"[bold]Duration:[/bold] {tracker._format_time(time.time() - tracker.start_time)}",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
