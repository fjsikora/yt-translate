#!/usr/bin/env python3
"""
Universal Video Translator - Cloud API Version

A FastAPI service that translates videos from 1700+ sites using cloud APIs:
- yt-dlp: Download video & extract audio (local)
- OpenAI Whisper API: Speech-to-text transcription
- Google Translate: Text translation
- Replicate API: Chatterbox TTS with voice cloning
- ffmpeg: Audio/video merging (local)

Designed for deployment on Railway, Render, or similar platforms.
No GPU required - all heavy lifting done by cloud APIs.
"""

import asyncio
import os
import re
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import httpx
import replicate
import yt_dlp
from deep_translator import GoogleTranslator
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

import db
from audio_processing import AudioSeparator, SpeakerDiarizer

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/yt-translate"))
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "600"))  # 10 minutes default
PRICE_PER_MINUTE_CENTS = int(os.getenv("PRICE_PER_MINUTE_CENTS", "50"))  # $0.50 per minute default
PREVIEW_DURATION_SECONDS = int(os.getenv("PREVIEW_DURATION_SECONDS", "60"))  # Free preview duration

# Rate limiting configuration
PREVIEW_RATE_LIMIT = int(os.getenv("PREVIEW_RATE_LIMIT", "5"))  # Max preview requests per window
PREVIEW_RATE_WINDOW = int(os.getenv("PREVIEW_RATE_WINDOW", "3600"))  # Window in seconds (1 hour)

# In-memory rate limiting storage (use Redis for production with multiple instances)
# Structure: {ip_address: [timestamp1, timestamp2, ...]}
preview_rate_limits: dict[str, list[float]] = defaultdict(list)

# Supported languages (Chatterbox multilingual model)
SUPPORTED_LANGUAGES = {
    "ar": "Arabic", "zh": "Chinese", "da": "Danish", "nl": "Dutch",
    "en": "English", "fi": "Finnish", "fr": "French", "de": "German",
    "el": "Greek", "he": "Hebrew", "hi": "Hindi", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "ms": "Malay", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "es": "Spanish",
    "sw": "Swahili", "sv": "Swedish", "tr": "Turkish",
}

# Google Translate language codes (may differ slightly)
GOOGLE_LANG_CODES = {
    "ar": "ar", "zh": "zh-CN", "da": "da", "nl": "nl", "en": "en",
    "fi": "fi", "fr": "fr", "de": "de", "el": "el", "he": "he",
    "hi": "hi", "it": "it", "ja": "ja", "ko": "ko", "ms": "ms",
    "no": "no", "pl": "pl", "pt": "pt", "ru": "ru", "es": "es",
    "sw": "sw", "sv": "sv", "tr": "tr",
}

# ISO 639-1 (2-letter) to ISO 639-2 (3-letter) language code mapping
# Used for ffmpeg subtitle metadata which requires 3-letter codes
ISO_639_1_TO_639_2 = {
    "ar": "ara",
    "zh": "chi",
    "da": "dan",
    "nl": "dut",
    "en": "eng",
    "fi": "fin",
    "fr": "fre",
    "de": "ger",
    "el": "gre",
    "he": "heb",
    "hi": "hin",
    "it": "ita",
    "ja": "jpn",
    "ko": "kor",
    "ms": "may",
    "no": "nor",
    "pl": "pol",
    "pt": "por",
    "ru": "rus",
    "es": "spa",
    "sw": "swa",
    "sv": "swe",
    "tr": "tur",
}

# In-memory job storage (use Redis for production)
jobs: dict[str, dict] = {}


# Pydantic models
class TranslateRequest(BaseModel):
    video_url: str  # Supports YouTube, Rumble, Vimeo, and 1700+ sites via yt-dlp
    target_language: str  # Language code (e.g., "es", "ja", "fr")


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int  # 0-100
    stage: Optional[str] = None
    error: Optional[str] = None
    output_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str


class VideoInfoRequest(BaseModel):
    video_url: str


class VideoInfoResponse(BaseModel):
    video_title: str
    duration_seconds: int
    thumbnail_url: Optional[str] = None
    price_quote: int  # Price in cents for full translation


class PreviewRequest(BaseModel):
    video_url: str  # URL of the video to preview
    target_language: str  # Language code (e.g., "es", "ja", "fr")
    session_id: str  # Anonymous session identifier for guest users


class PreviewResponse(BaseModel):
    preview_id: str  # UUID of the created preview job
    status: str  # "pending", "processing", "completed", "failed"
    message: str  # Human-readable status message


class PreviewStatusResponse(BaseModel):
    preview_id: str  # UUID of the preview job
    status: str  # "pending", "processing", "completed", "failed"
    progress: int  # 0-100
    stage: Optional[str] = None  # Current processing stage
    error: Optional[str] = None  # Error message if failed
    preview_url: Optional[str] = None  # Signed URL when completed
    price_quote: Optional[int] = None  # Price in cents for full translation


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    user_id: str
    email: str
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class CheckoutRequest(BaseModel):
    preview_id: str  # UUID of the preview job to purchase full translation for


class CheckoutResponse(BaseModel):
    checkout_url: str  # Stripe Checkout URL for redirect
    translation_job_id: str  # UUID of the created translation job


class TranslationJobStatusResponse(BaseModel):
    job_id: str  # UUID of the translation job
    status: str  # "pending", "processing", "completed", "failed"
    progress: int  # 0-100
    stage: Optional[str] = None  # Current processing stage
    error: Optional[str] = None  # Error message if failed
    download_url: Optional[str] = None  # Signed URL when completed


# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check API keys
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set")
    if not REPLICATE_API_TOKEN:
        print("WARNING: REPLICATE_API_TOKEN not set")

    yield

    # Shutdown - cleanup old files (optional)
    pass


app = FastAPI(
    title="Universal Video Translator API",
    description="Translate videos from YouTube, Rumble, Vimeo, and 1700+ sites to other languages using AI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Helper Functions ---

def update_job(job_id: str, **kwargs):
    """Update job status."""
    if job_id in jobs:
        jobs[job_id].update(kwargs)
        jobs[job_id]["updated_at"] = time.time()


async def download_video(
    url: str,
    job_id: str,
    duration_limit: Optional[int] = None
) -> tuple[Path, Path, dict]:
    """
    Download video from any supported site and extract audio.

    Args:
        url: Video URL (supports all yt-dlp sites)
        job_id: Unique job identifier for progress tracking
        duration_limit: Optional maximum duration in seconds. If provided,
                        both video and audio will be trimmed to this length.
                        Used for preview generation (e.g., 60 seconds).

    Returns:
        Tuple of (video_path, audio_path, info_dict)
    """
    update_job(job_id, stage="download", progress=5)

    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Get video info first
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info['id']
        title = info.get('title', video_id)
        duration = info.get('duration', 0)

        # For duration-limited downloads, only check against limit if no duration_limit
        # (preview clips don't need to respect MAX_VIDEO_DURATION)
        if duration_limit is None and duration > MAX_VIDEO_DURATION:
            raise ValueError(f"Video too long ({duration}s). Maximum is {MAX_VIDEO_DURATION}s.")

    update_job(job_id, progress=10)

    video_path = job_dir / f"{video_id}.mp4"
    audio_path = job_dir / f"{video_id}.wav"

    # Download video
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(job_dir / f"{video_id}.%(ext)s"),
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find downloaded video
    for ext in ['mp4', 'mkv', 'webm']:
        p = job_dir / f"{video_id}.{ext}"
        if p.exists():
            video_path = p
            break

    update_job(job_id, progress=20)

    # If duration_limit is specified, trim the video to that duration using ffmpeg
    if duration_limit is not None and duration_limit > 0:
        trimmed_video_path = job_dir / f"{video_id}_trimmed.mp4"
        subprocess.run([
            'ffmpeg', '-i', str(video_path),
            '-t', str(duration_limit),
            '-c:v', 'copy', '-c:a', 'copy',
            '-y', str(trimmed_video_path)
        ], capture_output=True, check=True)

        # Replace original with trimmed version
        video_path.unlink()  # Delete original
        trimmed_video_path.rename(video_path)  # Rename trimmed to original name

    update_job(job_id, progress=22)

    # Build ffmpeg command for audio extraction
    ffmpeg_audio_cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
    ]

    # Add duration limit for audio extraction if specified
    if duration_limit is not None and duration_limit > 0:
        ffmpeg_audio_cmd.extend(['-t', str(duration_limit)])

    ffmpeg_audio_cmd.extend(['-y', str(audio_path)])

    # Extract audio (with optional duration limit)
    subprocess.run(ffmpeg_audio_cmd, capture_output=True, check=True)

    update_job(job_id, progress=25)

    # Return actual duration (limited if applicable)
    actual_duration = min(duration, duration_limit) if duration_limit else duration

    return video_path, audio_path, {"title": title, "duration": actual_duration, "video_id": video_id}


async def transcribe_audio(audio_path: Path, job_id: str) -> list[dict]:
    """
    Transcribe audio using OpenAI Whisper API with segment timestamps.

    Returns segments in the same format as the local transcribe_audio():
    [{"start": float, "end": float, "text": str, "duration": float}, ...]
    """
    update_job(job_id, stage="transcribe", progress=30)

    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    update_job(job_id, progress=45)

    # Parse response to extract segments with timestamps
    # verbose_json response has: text, language, duration, segments[]
    # Each segment has: id, seek, start, end, text, tokens, temperature, etc.
    segments = []
    for seg in getattr(transcript, "segments", []) or []:
        # Handle both dict and object access patterns
        if isinstance(seg, dict):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "").strip()
        else:
            start = getattr(seg, "start", 0.0)
            end = getattr(seg, "end", 0.0)
            text = getattr(seg, "text", "").strip()

        segments.append({
            "start": start,
            "end": end,
            "text": text,
            "duration": end - start
        })

    return segments


async def translate_text(text: str, target_lang: str, job_id: str) -> str:
    """Translate text using Google Translate."""
    update_job(job_id, stage="translate", progress=50)

    google_code = GOOGLE_LANG_CODES.get(target_lang, target_lang)

    # Split into chunks if text is too long (Google has limits)
    max_chunk_size = 4500
    if len(text) <= max_chunk_size:
        translated = GoogleTranslator(source='auto', target=google_code).translate(text)
    else:
        # Translate in chunks
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        translated_chunks = []
        for chunk in chunks:
            translated_chunk = GoogleTranslator(source='auto', target=google_code).translate(chunk)
            translated_chunks.append(translated_chunk)
        translated = " ".join(translated_chunks)

    update_job(job_id, progress=55)
    return translated


async def translate_segments(segments: list[dict], target_lang: str, job_id: str) -> list[dict]:
    """
    Translate each segment while preserving timing info.

    Returns segments in the same format as the local translate_segments():
    [{"start": float, "end": float, "duration": float, "original_text": str, "translated_text": str}, ...]
    """
    update_job(job_id, stage="translate", progress=50)

    google_code = GOOGLE_LANG_CODES.get(target_lang, target_lang)

    translated_segments = []
    total = len(segments)

    for i, seg in enumerate(segments):
        text = seg.get("text", "")

        if text:
            try:
                translated_text = GoogleTranslator(source='auto', target=google_code).translate(text)
            except Exception:
                translated_text = text  # Fallback to original if translation fails
        else:
            translated_text = ""

        translated_segments.append({
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "duration": seg.get("duration", 0.0),
            "original_text": text,
            "translated_text": translated_text
        })

    update_job(job_id, progress=55)
    return translated_segments


async def synthesize_speech(
    text: str,
    audio_prompt_path: Path,
    target_lang: str,
    job_id: str,
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5
) -> Path:
    """
    Generate speech using Replicate Chatterbox API with voice cloning.

    Args:
        text: The text to synthesize
        audio_prompt_path: Path to source audio for voice cloning
        target_lang: Target language code (e.g., "es", "ja")
        job_id: Job ID for tracking and storage
        cfg_weight: CFG/Pace weight (0.0-1.0, higher = slower/more stable)
        exaggeration: Voice exaggeration (0.0-1.0, higher = more expressive)

    Returns:
        Path to the synthesized audio file
    """
    update_job(job_id, stage="synthesize", progress=60)

    job_dir = OUTPUT_DIR / job_id
    output_audio = job_dir / "translated_audio.wav"

    # Extract a 10-second voice sample for cloning (24kHz mono for Chatterbox)
    voice_sample = job_dir / "voice_sample.wav"
    subprocess.run([
        'ffmpeg', '-i', str(audio_prompt_path),
        '-t', '10',
        '-ar', '24000',  # Chatterbox requires 24kHz
        '-ac', '1',       # Mono
        '-y', str(voice_sample)
    ], capture_output=True, check=True)

    update_job(job_id, progress=62)

    # Upload voice sample to Supabase Storage for Replicate API access
    try:
        storage_path = db.upload_voice_sample(job_id, str(voice_sample))
        voice_sample_url = db.get_voice_sample_signed_url(storage_path, expires_in=3600)
    except Exception as e:
        raise RuntimeError(f"Failed to upload voice sample: {e}")

    update_job(job_id, progress=65)

    # Call Replicate Chatterbox API with voice cloning
    try:
        output = replicate.run(
            "resemble-ai/chatterbox-multilingual",
            input={
                "text": text,
                "language_id": target_lang,
                "audio_prompt": voice_sample_url,
                "cfg_weight": cfg_weight,
                "exaggeration": exaggeration,
            }
        )

        # Download the output audio
        if isinstance(output, str):
            # Output is a URL
            async with httpx.AsyncClient() as client:
                response = await client.get(output)
                with open(output_audio, "wb") as f:
                    f.write(response.content)
        else:
            # Output might be file-like or iterator
            with open(output_audio, "wb") as f:
                for chunk in output:
                    f.write(chunk)

    except Exception as e:
        raise RuntimeError(f"TTS synthesis failed: {e}")

    update_job(job_id, progress=85)
    return output_audio


async def separate_audio(audio_path: Path, job_id: str) -> tuple[Path, Path]:
    """
    Separate audio into vocals and background using Demucs.

    Args:
        audio_path: Path to the input audio file
        job_id: Job ID for progress tracking

    Returns:
        Tuple of (vocals_path, background_path)
    """
    update_job(job_id, stage="separate", progress=26)

    job_dir = OUTPUT_DIR / job_id

    try:
        separator = AudioSeparator()
        vocals, background, sample_rate = separator.separate(audio_path)

        update_job(job_id, progress=28)

        vocals_path, background_path = separator.save_separated(
            vocals, background, sample_rate, job_dir
        )

        update_job(job_id, progress=30)

        return vocals_path, background_path

    except Exception as e:
        # Fallback: if separation fails, use original audio as vocals and create empty background
        print(f"Warning: Audio separation failed, using original audio: {e}")
        return audio_path, audio_path


async def diarize_speakers(
    vocals_path: Path,
    job_id: str
) -> tuple[list[dict], dict[str, str]]:
    """
    Run speaker diarization on vocals audio and extract per-speaker voice samples.

    Uses pyannote.audio to identify different speakers, then extracts voice
    samples for each speaker and uploads them to Supabase Storage.

    Args:
        vocals_path: Path to the vocals audio file
        job_id: Job ID for progress tracking and storage path

    Returns:
        Tuple of (diarization_segments, speaker_voice_urls) where:
        - diarization_segments: List of segment dicts with speaker labels and timestamps
        - speaker_voice_urls: Dict mapping speaker ID to signed URL for voice sample

    Note:
        Falls back to empty results if diarization fails (allows pipeline to continue
        with single-speaker synthesis).
    """
    update_job(job_id, stage="diarize", progress=31)

    job_dir = OUTPUT_DIR / job_id
    diarization_segments: list[dict] = []
    speaker_voice_urls: dict[str, str] = {}

    try:
        # Initialize diarizer
        diarizer = SpeakerDiarizer()

        # Run diarization on vocals
        diarization_segments = diarizer.diarize(vocals_path)

        update_job(job_id, progress=35)

        if not diarization_segments:
            print(f"Warning: No speakers detected in audio for job {job_id}")
            return [], {}

        # Extract per-speaker voice samples (24kHz mono WAV for Chatterbox)
        speaker_samples = diarizer.extract_speaker_samples(
            audio_path=vocals_path,
            output_dir=job_dir,
            segments=diarization_segments
        )

        update_job(job_id, progress=38)

        # Upload each speaker's voice sample to Supabase Storage
        for speaker_id, sample_path in speaker_samples.items():
            try:
                # Upload voice sample with speaker ID in the path
                storage_path = db.upload_voice_sample(
                    job_id=job_id,
                    file_path=str(sample_path),
                    speaker_id=speaker_id
                )

                # Get signed URL for Replicate API access
                signed_url = db.get_voice_sample_signed_url(
                    storage_path=storage_path,
                    expires_in=3600  # 1 hour expiration
                )

                speaker_voice_urls[speaker_id] = signed_url

            except Exception as e:
                print(f"Warning: Failed to upload voice sample for {speaker_id}: {e}")
                continue

        update_job(job_id, progress=40)

        return diarization_segments, speaker_voice_urls

    except Exception as e:
        # Fallback: if diarization fails, return empty results
        # The pipeline will continue with single-speaker synthesis
        print(f"Warning: Speaker diarization failed for job {job_id}: {e}")
        return [], {}


async def synthesize_segments_multi_speaker(
    translated_segments: list[dict],
    diarization_segments: list[dict],
    speaker_voice_urls: dict[str, str],
    target_lang: str,
    job_id: str,
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5
) -> Path:
    """
    Synthesize speech for each segment using speaker-matched voices.

    Maps each transcription segment to its speaker via diarization timestamps,
    then uses the appropriate voice sample URL for Replicate Chatterbox generation.

    Audio plays at natural 1x speed. Segment start times are adjusted to prevent
    overlap - if a segment hasn't finished, the next one waits for it to complete.

    Args:
        translated_segments: Translated text segments with start/end times and translated_text
        diarization_segments: Speaker diarization results with speaker labels and timestamps
        speaker_voice_urls: Dict mapping speaker ID to signed URL for voice sample
        target_lang: Target language code for Chatterbox (e.g., "es", "ja")
        job_id: Job ID for tracking and storage
        cfg_weight: CFG/Pace weight (0.0-1.0, higher = slower/more stable)
        exaggeration: Voice exaggeration (0.0-1.0, higher = more expressive)

    Returns:
        Path to the generated combined audio file
    """
    import wave
    import struct
    import numpy as np

    update_job(job_id, stage="synthesize", progress=60)

    job_dir = OUTPUT_DIR / job_id
    output_audio_path = job_dir / "translated_audio.wav"

    # Get fallback voice URL (first available)
    if not speaker_voice_urls:
        raise RuntimeError("No speaker voice URLs available for synthesis")

    fallback_speaker = next(iter(speaker_voice_urls))
    fallback_voice_url = speaker_voice_urls[fallback_speaker]

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

    def read_wav_file(wav_path: Path) -> tuple[int, np.ndarray]:
        """Read a WAV file and return sample rate and float32 audio data."""
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
        """Write float32 audio data to a WAV file as 16-bit."""
        # Convert to 16-bit integer
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        if orig_sr == target_sr:
            return audio
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)

    # Process each segment and collect audio with original start times
    total_segments = len(translated_segments)
    segment_audios: list[tuple[float, np.ndarray]] = []  # (original_start, audio)

    # Target sample rate (Chatterbox typically outputs 24kHz)
    target_sample_rate = 24000

    for i, seg in enumerate(translated_segments):
        text = seg.get("translated_text", "")

        if not text:
            # Skip empty segments
            continue

        # Match this segment to a speaker
        speaker = match_segment_to_speaker(seg["start"], seg["end"])

        # Get voice URL for this speaker (with fallback)
        voice_url = speaker_voice_urls.get(speaker, fallback_voice_url)

        # Update progress (60-85% range for synthesis)
        progress = 60 + int(25 * i / max(1, total_segments))
        update_job(job_id, progress=progress)

        # Call Replicate Chatterbox API for this segment
        try:
            output = replicate.run(
                "resemble-ai/chatterbox-multilingual",
                input={
                    "text": text,
                    "language_id": target_lang,
                    "audio_prompt": voice_url,
                    "cfg_weight": cfg_weight,
                    "exaggeration": exaggeration,
                }
            )

            # Download the segment audio
            segment_audio_path = job_dir / f"segment_{i}.wav"

            if isinstance(output, str):
                # Output is a URL
                async with httpx.AsyncClient() as client:
                    response = await client.get(output)
                    with open(segment_audio_path, "wb") as f:
                        f.write(response.content)
            else:
                # Output might be file-like or iterator
                with open(segment_audio_path, "wb") as f:
                    for chunk in output:
                        f.write(chunk)

            # Read the audio data using standard library
            sr, audio_np = read_wav_file(segment_audio_path)

            # Resample if needed
            if sr != target_sample_rate:
                audio_np = resample_audio(audio_np, sr, target_sample_rate)

            segment_audios.append((seg["start"], audio_np))

            # Clean up segment file
            segment_audio_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"Warning: Failed to synthesize segment {i} for job {job_id}: {e}")
            continue

    if not segment_audios:
        raise RuntimeError("No segments were successfully synthesized")

    # Combine all segments with adjusted timing to prevent overlap
    # Each segment starts at max(original_start, previous_segment_end)
    positioned_audios: list[tuple[float, np.ndarray]] = []
    current_end_time = 0.0

    for original_start, audio in segment_audios:
        audio_duration = len(audio) / target_sample_rate

        # Start at the later of: original timestamp or when previous segment ends
        actual_start = max(original_start, current_end_time)
        positioned_audios.append((actual_start, audio))

        # Update end time for next segment
        current_end_time = actual_start + audio_duration

    # Calculate actual output length
    total_duration = max(
        translated_segments[-1].get("end", 0.0) if translated_segments else 0.0,
        current_end_time
    )
    output_length = int(total_duration * target_sample_rate)
    output_audio = np.zeros(output_length)

    # Place audio at calculated positions
    for start_time, audio in positioned_audios:
        start_sample = int(start_time * target_sample_rate)
        end_sample = start_sample + len(audio)

        # Ensure we don't overflow
        if end_sample > output_length:
            audio = audio[:output_length - start_sample]

        # Place audio
        if start_sample < output_length:
            output_audio[start_sample:start_sample + len(audio)] = audio

    # Normalize to prevent clipping
    max_val = np.max(np.abs(output_audio))
    if max_val > 0.99:
        output_audio = output_audio * 0.99 / max_val

    # Save the combined audio
    write_wav_file(output_audio_path, target_sample_rate, output_audio)

    update_job(job_id, progress=85)
    return output_audio_path


def get_iso_639_2_code(lang_code: str) -> str:
    """
    Convert ISO 639-1 (2-letter) language code to ISO 639-2 (3-letter) code.

    Args:
        lang_code: 2-letter or 3-letter language code

    Returns:
        3-letter ISO 639-2 language code for ffmpeg metadata
    """
    if len(lang_code) == 3:
        return lang_code
    return ISO_639_1_TO_639_2.get(lang_code, "eng")


def generate_srt(segments: list[dict], output_path: Optional[Path] = None, job_id: Optional[str] = None) -> Path:
    """
    Generate SRT subtitle file from translated segments.

    Args:
        segments: List of translated segments with start, end, and translated_text fields
        output_path: Output path for the SRT file. If None, uses job_dir/"subtitles.srt"
        job_id: Job ID for determining output directory (required if output_path is None)

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
        if job_id is None:
            raise ValueError("Either output_path or job_id must be provided")
        output_path = OUTPUT_DIR / job_id / "subtitles.srt"

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
    """
    Mix translated speech with original background audio using ffmpeg.

    Args:
        speech_path: Path to the translated speech WAV file (100% volume)
        background_path: Path to the background audio WAV file
        output_path: Output path for the mixed audio. If None, uses speech_path directory/"mixed_audio.wav"
        background_volume: Volume level for background audio (0.0 to 1.0, default 0.3 = 30%)

    Returns:
        Path to the mixed audio file

    The output duration matches the speech track. If background is shorter,
    it's padded with silence. If background is longer, it's trimmed.
    """
    if output_path is None:
        output_path = speech_path.parent / "mixed_audio.wav"

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


async def merge_audio_video(
    video_path: Path,
    audio_path: Path,
    job_id: str,
    title: str,
    subtitle_path: Optional[Path] = None,
    subtitle_lang: str = "eng"
) -> Path:
    """
    Merge translated audio with original video, optionally embedding subtitles.

    Args:
        video_path: Path to the video file
        audio_path: Path to the translated audio file
        job_id: Job ID for progress tracking
        title: Video title for generating output filename
        subtitle_path: Optional path to SRT subtitle file
        subtitle_lang: ISO 639-2 language code for subtitle metadata (default: "eng")

    Returns:
        Path to the merged output video file

    Subtitles are embedded as soft subs (toggleable, not burned in) using the
    mov_text codec for MP4 compatibility.
    """
    update_job(job_id, stage="merge", progress=90)

    job_dir = OUTPUT_DIR / job_id
    safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:40]
    output_path = job_dir / f"{safe_title}_translated.mp4"

    # Build ffmpeg command based on whether subtitles are provided
    if subtitle_path and subtitle_path.exists():
        # With subtitles: include subtitle input and mapping
        subprocess.run([
            'ffmpeg',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-i', str(subtitle_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-c:s', 'mov_text',  # Soft subtitles codec for MP4
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-map', '2:0',
            '-metadata:s:s:0', f'language={subtitle_lang}',  # Set subtitle language
            '-shortest',
            '-y',
            str(output_path)
        ], capture_output=True, check=True)
    else:
        # Without subtitles: original simple merge
        subprocess.run([
            'ffmpeg',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-c:v', 'copy',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y',
            str(output_path)
        ], capture_output=True, check=True)

    update_job(job_id, progress=100)
    return output_path


async def process_translation(job_id: str, video_url: str, target_language: str):
    """
    Main translation pipeline.

    Pipeline: download -> separate -> diarize -> transcribe -> translate -> TTS -> mix -> subtitles -> merge
    """
    try:
        update_job(job_id, status="processing", progress=0)

        # 1. Download video
        video_path, audio_path, info = await download_video(video_url, job_id)

        # 2. Separate audio into vocals and background
        vocals_path, background_path = await separate_audio(audio_path, job_id)

        # 3. Run speaker diarization on vocals
        diarization_segments, speaker_voice_urls = await diarize_speakers(vocals_path, job_id)

        # 4. Transcribe vocals (cleaner input without background noise)
        segments = await transcribe_audio(vocals_path, job_id)

        # 5. Translate segments (preserves timestamps)
        translated_segments = await translate_segments(segments, target_language, job_id)

        # 6. Synthesize (TTS) - use multi-speaker if diarization available, else single speaker
        if speaker_voice_urls and diarization_segments:
            # Multi-speaker synthesis using per-segment voice matching
            new_audio = await synthesize_segments_multi_speaker(
                translated_segments=translated_segments,
                diarization_segments=diarization_segments,
                speaker_voice_urls=speaker_voice_urls,
                target_lang=target_language,
                job_id=job_id
            )
        else:
            # Fallback: single speaker synthesis (combine all text)
            translated_text = " ".join(
                seg["translated_text"] for seg in translated_segments if seg.get("translated_text")
            )
            new_audio = await synthesize_speech(translated_text, vocals_path, target_language, job_id)

        # 7. Mix translated speech with original background audio (30% volume)
        update_job(job_id, stage="mix", progress=86)
        # Only mix if background_path is different from audio_path (separation succeeded)
        if background_path != audio_path:
            mixed_audio = mix_audio_with_background(new_audio, background_path)
        else:
            mixed_audio = new_audio

        # 8. Generate SRT subtitles from translated segments
        update_job(job_id, stage="subtitles", progress=88)
        subtitle_path = generate_srt(translated_segments, job_id=job_id)
        subtitle_lang = get_iso_639_2_code(target_language)

        # 9. Merge audio and video with embedded subtitles
        output = await merge_audio_video(
            video_path, mixed_audio, job_id, info["title"],
            subtitle_path=subtitle_path, subtitle_lang=subtitle_lang
        )

        # Update job as completed
        update_job(
            job_id,
            status="completed",
            progress=100,
            stage="done",
            output_file=str(output),
            output_url=f"/download/{job_id}"
        )

    except Exception as e:
        update_job(
            job_id,
            status="failed",
            error=str(e),
            stage="error"
        )
        raise


def calculate_price_cents(duration_seconds: int) -> int:
    """
    Calculate price in cents based on video duration.

    Price is calculated per minute, rounded up.
    Minimum price is 1 minute.
    """
    minutes = max(1, (duration_seconds + 59) // 60)  # Round up to nearest minute
    return minutes * PRICE_PER_MINUTE_CENTS


def check_preview_rate_limit(client_ip: str) -> tuple[bool, int]:
    """
    Check if the client IP has exceeded the preview rate limit.

    Implements a sliding window rate limiter that allows PREVIEW_RATE_LIMIT
    requests per PREVIEW_RATE_WINDOW seconds.

    Args:
        client_ip: The client's IP address

    Returns:
        Tuple of (is_allowed, retry_after_seconds)
        - is_allowed: True if request is allowed, False if rate limited
        - retry_after_seconds: Seconds to wait before retry (0 if allowed)
    """
    current_time = time.time()
    window_start = current_time - PREVIEW_RATE_WINDOW

    # Get existing timestamps for this IP and filter to current window
    timestamps = preview_rate_limits[client_ip]
    valid_timestamps = [ts for ts in timestamps if ts > window_start]

    # Update the stored timestamps (cleanup old ones)
    preview_rate_limits[client_ip] = valid_timestamps

    # Check if rate limit exceeded
    if len(valid_timestamps) >= PREVIEW_RATE_LIMIT:
        # Calculate retry-after: time until oldest request expires from window
        oldest_timestamp = min(valid_timestamps)
        retry_after = int(oldest_timestamp + PREVIEW_RATE_WINDOW - current_time) + 1
        return False, max(1, retry_after)

    # Request allowed - record this timestamp
    preview_rate_limits[client_ip].append(current_time)
    return True, 0


def get_client_ip(request: Request) -> str:
    """
    Extract the client IP address from a request.

    Handles common proxy headers (X-Forwarded-For, X-Real-IP) and falls back
    to the direct client IP if no proxy headers are present.

    Args:
        request: FastAPI Request object

    Returns:
        Client IP address as string
    """
    # Check for proxy headers (common when behind load balancers/CDNs)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs: client, proxy1, proxy2...
        # The first one is the original client
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fall back to direct client IP
    if request.client:
        return request.client.host

    return "unknown"


# --- API Endpoints ---

@app.get("/")
async def root():
    """Redirect to frontend."""
    return RedirectResponse(url="/static/index.html")


@app.get("/success")
async def success_page(request: Request):
    """Redirect to payment success page, preserving query parameters."""
    query_string = str(request.query_params)
    if query_string:
        return RedirectResponse(url=f"/static/success.html?{query_string}")
    return RedirectResponse(url="/static/success.html")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Railway/load balancers."""
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/languages")
async def list_languages():
    """List supported target languages."""
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/video-info", response_model=VideoInfoResponse)
async def get_video_info(request: VideoInfoRequest):
    """
    Get video information and price quote.

    Extracts video title, duration, and thumbnail from the provided URL
    and calculates a price quote for full translation.
    """
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(request.video_url, download=False)

            if info is None:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract video information from the provided URL"
                )

            title = info.get('title', 'Unknown Title')
            duration = info.get('duration')

            if duration is None:
                raise HTTPException(
                    status_code=400,
                    detail="Could not determine video duration. The URL may not point to a valid video."
                )

            # Get thumbnail (yt-dlp provides various thumbnail options)
            thumbnail = info.get('thumbnail')
            if not thumbnail:
                thumbnails = info.get('thumbnails', [])
                if thumbnails:
                    thumbnail = thumbnails[-1].get('url')

            price_cents = calculate_price_cents(duration)

            return VideoInfoResponse(
                video_title=title,
                duration_seconds=duration,
                thumbnail_url=thumbnail,
                price_quote=price_cents
            )

    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video URL or video not accessible: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract video information: {str(e)}"
        )


async def process_preview(preview_id: str, video_url: str, target_language: str):
    """
    Background task to process a preview translation (first 60 seconds only).

    Uses PREVIEW_DURATION_SECONDS to limit the video/audio duration.
    Pipeline: download -> separate -> diarize -> transcribe -> translate -> TTS -> mix -> subtitles -> merge -> upload
    """
    try:
        # Initialize job tracking in memory for helper functions
        jobs[preview_id] = {
            "job_id": preview_id,
            "status": "processing",
            "progress": 0,
            "stage": "initializing",
        }

        # Update status in database
        db.update_preview_job(preview_id, status="processing", progress=0, stage="initializing")

        # 1. Download first PREVIEW_DURATION_SECONDS of video
        db.update_preview_job(preview_id, stage="download", progress=5)
        video_path, audio_path, info = await download_video(
            video_url, preview_id, duration_limit=PREVIEW_DURATION_SECONDS
        )
        db.update_preview_job(preview_id, progress=25)

        # 2. Separate audio into vocals and background
        db.update_preview_job(preview_id, stage="separate", progress=26)
        vocals_path, background_path = await separate_audio(audio_path, preview_id)
        db.update_preview_job(preview_id, progress=30)

        # 3. Run speaker diarization on vocals
        db.update_preview_job(preview_id, stage="diarize", progress=31)
        diarization_segments, speaker_voice_urls = await diarize_speakers(vocals_path, preview_id)
        db.update_preview_job(preview_id, progress=40)

        # 4. Transcribe vocals (cleaner input without background noise)
        db.update_preview_job(preview_id, stage="transcribe", progress=41)
        segments = await transcribe_audio(vocals_path, preview_id)
        db.update_preview_job(preview_id, progress=48)

        # 5. Translate segments (preserves timestamps)
        db.update_preview_job(preview_id, stage="translate", progress=50)
        translated_segments = await translate_segments(segments, target_language, preview_id)
        db.update_preview_job(preview_id, progress=55)

        # 6. Synthesize speech (TTS) - use multi-speaker if diarization available
        db.update_preview_job(preview_id, stage="synthesize", progress=60)
        if speaker_voice_urls and diarization_segments:
            # Multi-speaker synthesis using per-segment voice matching
            new_audio = await synthesize_segments_multi_speaker(
                translated_segments=translated_segments,
                diarization_segments=diarization_segments,
                speaker_voice_urls=speaker_voice_urls,
                target_lang=target_language,
                job_id=preview_id
            )
        else:
            # Fallback: single speaker synthesis (combine all text)
            translated_text = " ".join(
                seg["translated_text"] for seg in translated_segments if seg.get("translated_text")
            )
            new_audio = await synthesize_speech(translated_text, vocals_path, target_language, preview_id)
        db.update_preview_job(preview_id, progress=82)

        # 7. Mix translated speech with original background audio (30% volume)
        db.update_preview_job(preview_id, stage="mix", progress=83)
        # Only mix if background_path is different from audio_path (separation succeeded)
        if background_path != audio_path:
            mixed_audio = mix_audio_with_background(new_audio, background_path)
        else:
            mixed_audio = new_audio
        db.update_preview_job(preview_id, progress=85)

        # 8. Generate SRT subtitles from translated segments
        db.update_preview_job(preview_id, stage="subtitles", progress=86)
        subtitle_path = generate_srt(translated_segments, job_id=preview_id)
        subtitle_lang = get_iso_639_2_code(target_language)
        db.update_preview_job(preview_id, progress=88)

        # 9. Merge translated audio with video (with embedded subtitles)
        db.update_preview_job(preview_id, stage="merge", progress=90)
        output_path = await merge_audio_video(
            video_path, mixed_audio, preview_id, info["title"],
            subtitle_path=subtitle_path, subtitle_lang=subtitle_lang
        )
        db.update_preview_job(preview_id, progress=95)

        # 10. Upload to Supabase Storage
        db.update_preview_job(preview_id, stage="upload", progress=95)
        storage_path = db.upload_preview_to_storage(preview_id, str(output_path))

        # 11. Update job as completed with file path
        db.update_preview_job(
            preview_id,
            status="completed",
            progress=100,
            stage="done",
            preview_file_path=storage_path
        )

        # Update in-memory job
        update_job(
            preview_id,
            status="completed",
            progress=100,
            stage="done",
            output_file=str(output_path)
        )

    except Exception as e:
        # Update both database and in-memory state
        db.update_preview_job(
            preview_id,
            status="failed",
            error_message=str(e),
            stage="error"
        )
        if preview_id in jobs:
            update_job(
                preview_id,
                status="failed",
                error=str(e),
                stage="error"
            )


@app.post("/preview", response_model=PreviewResponse)
async def create_preview(
    preview_request: PreviewRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Start a free preview translation job (first 60 seconds only).

    Creates a preview job record in the database and starts background processing.
    Guest users are identified by session_id.

    Rate limited to PREVIEW_RATE_LIMIT requests per PREVIEW_RATE_WINDOW per IP address.
    This does not affect authenticated users' paid translation jobs.
    """
    # Check rate limit by IP address
    client_ip = get_client_ip(request)
    is_allowed, retry_after = check_preview_rate_limit(client_ip)

    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {PREVIEW_RATE_LIMIT} preview requests per hour. Try again later.",
            headers={"Retry-After": str(retry_after)}
        )

    # Validate language
    if preview_request.target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {preview_request.target_language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    # Validate session_id is not empty
    if not preview_request.session_id or not preview_request.session_id.strip():
        raise HTTPException(
            status_code=400,
            detail="session_id is required"
        )

    # Validate video URL by attempting to extract info
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(preview_request.video_url, download=False)
            if info is None:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract video information from the provided URL"
                )
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video URL or video not accessible: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to validate video URL: {str(e)}"
        )

    # Create preview job in Supabase
    try:
        preview_job = db.create_preview_job(
            session_id=preview_request.session_id.strip(),
            video_url=preview_request.video_url,
            target_language=preview_request.target_language
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create preview job: {str(e)}"
        )

    preview_id = preview_job.get("id")
    if not preview_id:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve preview job ID"
        )

    # Start background processing
    background_tasks.add_task(
        process_preview,
        preview_id,
        preview_request.video_url,
        preview_request.target_language
    )

    return PreviewResponse(
        preview_id=preview_id,
        status="pending",
        message="Preview job created. Processing will begin shortly."
    )


@app.get("/preview/{preview_id}", response_model=PreviewStatusResponse)
async def get_preview_status(
    preview_id: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """
    Get the status of a preview job.

    Access control:
    - Guests can access by providing matching session_id as query param
    - Logged-in users can access by providing matching user_id as query param
    - At least one of session_id or user_id must be provided

    Returns:
    - status, progress, stage for tracking
    - preview_url and price_quote when completed
    """
    # Validate that at least one identifier is provided
    if not session_id and not user_id:
        raise HTTPException(
            status_code=400,
            detail="Either session_id or user_id must be provided"
        )

    # Get the preview job from database
    preview_job = db.get_preview_job(preview_id)

    if not preview_job:
        raise HTTPException(status_code=404, detail="Preview job not found")

    # Access control: verify ownership
    job_session_id = preview_job.get("session_id")
    job_user_id = preview_job.get("user_id")

    # Check if the requester owns this preview
    has_access = False
    if session_id and job_session_id == session_id:
        has_access = True
    if user_id and job_user_id == user_id:
        has_access = True

    if not has_access:
        raise HTTPException(
            status_code=403,
            detail="Access denied. You do not have permission to view this preview."
        )

    # Build response
    status = preview_job.get("status", "pending")
    progress = preview_job.get("progress", 0)
    stage = preview_job.get("stage")
    error = preview_job.get("error_message")

    preview_url = None
    price_quote = None

    # When completed, include preview_url and price_quote
    if status == "completed":
        preview_file_path = preview_job.get("preview_file_path")
        if preview_file_path:
            try:
                preview_url = db.get_preview_signed_url(preview_file_path, expires_in=3600)
            except Exception:
                # If signed URL generation fails, still return status
                pass

        # Calculate price quote from video URL
        video_url = preview_job.get("video_url")
        if video_url:
            try:
                with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    if info:
                        duration = info.get('duration', 0)
                        price_quote = calculate_price_cents(duration)
            except Exception:
                # If price calculation fails, continue without it
                pass

    return PreviewStatusResponse(
        preview_id=preview_id,
        status=status,
        progress=progress,
        stage=stage,
        error=error,
        preview_url=preview_url,
        price_quote=price_quote
    )


@app.get("/preview/{preview_id}/video")
async def get_preview_video(
    preview_id: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """
    Get the preview video file via signed URL redirect.

    Generates a signed URL from Supabase Storage that expires after 1 hour.

    Access control:
    - Guests can access by providing matching session_id as query param
    - Logged-in users can access by providing matching user_id as query param
    - At least one of session_id or user_id must be provided

    Returns:
    - Redirect to signed URL for the preview video

    Raises:
    - 400: Missing session_id/user_id
    - 403: Access denied (not the preview owner)
    - 404: Preview not found or not completed
    """
    # Validate that at least one identifier is provided
    if not session_id and not user_id:
        raise HTTPException(
            status_code=400,
            detail="Either session_id or user_id must be provided"
        )

    # Get the preview job from database
    preview_job = db.get_preview_job(preview_id)

    if not preview_job:
        raise HTTPException(status_code=404, detail="Preview job not found")

    # Access control: verify ownership
    job_session_id = preview_job.get("session_id")
    job_user_id = preview_job.get("user_id")

    # Check if the requester owns this preview
    has_access = False
    if session_id and job_session_id == session_id:
        has_access = True
    if user_id and job_user_id == user_id:
        has_access = True

    if not has_access:
        raise HTTPException(
            status_code=403,
            detail="Access denied. You do not have permission to view this preview."
        )

    # Verify preview is completed
    status = preview_job.get("status")
    if status != "completed":
        raise HTTPException(
            status_code=404,
            detail=f"Preview video not available. Current status: {status}"
        )

    # Get the file path and generate signed URL
    preview_file_path = preview_job.get("preview_file_path")
    if not preview_file_path:
        raise HTTPException(
            status_code=404,
            detail="Preview file not found"
        )

    try:
        # Generate signed URL with 1 hour expiration
        signed_url = db.get_preview_signed_url(preview_file_path, expires_in=3600)
        if not signed_url:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate video URL"
            )
        return RedirectResponse(url=signed_url, status_code=302)
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid storage path: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate video URL: {str(e)}"
        )


# --- Authentication Endpoints ---

# Email validation regex pattern (RFC 5322 simplified)
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Password requirements
MIN_PASSWORD_LENGTH = 8


def validate_email(email: str) -> bool:
    """Validate email format using regex."""
    return bool(EMAIL_REGEX.match(email))


def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password strength.

    Requirements:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    return True, ""


# --- Authentication Middleware ---

# HTTP Bearer security scheme for JWT tokens
security = HTTPBearer(auto_error=False)


class CurrentUser(BaseModel):
    """Represents the currently authenticated user."""
    user_id: str
    email: str


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> CurrentUser:
    """
    FastAPI dependency to get the current authenticated user from JWT token.

    Extracts the Bearer token from the Authorization header, validates it
    using Supabase Auth, and returns the user information.

    Args:
        credentials: HTTP Bearer credentials from Authorization header

    Returns:
        CurrentUser with user_id and email

    Raises:
        HTTPException 401: If token is missing or invalid
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = credentials.credentials

    try:
        user_info = db.verify_jwt(token)
        return CurrentUser(
            user_id=user_info["user_id"],
            email=user_info["email"]
        )
    except ValueError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[CurrentUser]:
    """
    FastAPI dependency to optionally get the current authenticated user.

    Similar to get_current_user but returns None instead of raising 401
    if no token is provided. Useful for endpoints that support both
    authenticated and guest access.

    Args:
        credentials: HTTP Bearer credentials from Authorization header

    Returns:
        CurrentUser with user_id and email, or None if not authenticated
    """
    if credentials is None:
        return None

    token = credentials.credentials

    try:
        user_info = db.verify_jwt(token)
        return CurrentUser(
            user_id=user_info["user_id"],
            email=user_info["email"]
        )
    except Exception:
        return None


@app.post("/auth/signup", response_model=AuthResponse)
async def signup(request: SignupRequest):
    """
    Create a new user account.

    Validates email format and password strength, creates user in Supabase Auth,
    and creates a profile record with tos_accepted_at timestamp.

    Returns:
        AuthResponse with user_id, email, and session tokens
    """
    # Validate email format
    if not validate_email(request.email):
        raise HTTPException(
            status_code=400,
            detail="Invalid email format"
        )

    # Validate password strength
    is_valid, error_msg = validate_password(request.password)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=error_msg
        )

    try:
        # Create user in Supabase Auth
        auth_result = db.signup_user(request.email, request.password)

        user_id = auth_result["user_id"]
        email = auth_result["email"]

        # Create profile record with tos_accepted_at
        try:
            db.create_profile(
                user_id=user_id,
                email=email,
                tos_accepted=True  # ToS acceptance is implicit in signup
            )
        except Exception as profile_error:
            # If profile creation fails, the user was still created in Auth
            # Log this but don't fail the signup
            print(f"Warning: Failed to create profile for user {user_id}: {profile_error}")

        # Return session tokens
        return AuthResponse(
            user_id=user_id,
            email=email,
            access_token=auth_result["access_token"],
            refresh_token=auth_result["refresh_token"]
        )

    except ValueError as e:
        # Validation errors from db.signup_user
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Signup failed: {str(e)}"
        )


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """
    Log in an existing user.

    Returns:
        AuthResponse with user_id, email, and session tokens

    Raises:
        401: Invalid credentials
    """
    try:
        # Authenticate with Supabase Auth
        auth_result = db.login_user(request.email, request.password)

        return AuthResponse(
            user_id=auth_result["user_id"],
            email=auth_result["email"],
            access_token=auth_result["access_token"],
            refresh_token=auth_result["refresh_token"]
        )

    except ValueError as e:
        # Invalid credentials
        raise HTTPException(
            status_code=401,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Login failed: {str(e)}"
        )


# Configuration for OAuth redirect URL
OAUTH_REDIRECT_URL = os.getenv("OAUTH_REDIRECT_URL", "http://localhost:8000/auth/callback")


@app.get("/auth/google")
async def google_oauth_redirect():
    """
    Redirect to Google OAuth for sign-in.

    Initiates the Google OAuth flow by redirecting the user to Google's
    authorization page. After successful authentication, the user will be
    redirected back to /auth/callback.

    Returns:
        RedirectResponse to Google OAuth URL
    """
    try:
        oauth_url = db.get_google_oauth_url(OAUTH_REDIRECT_URL)
        return RedirectResponse(url=oauth_url, status_code=302)

    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate Google OAuth: {str(e)}"
        )


@app.get("/auth/callback")
async def oauth_callback(code: Optional[str] = None, error: Optional[str] = None):
    """
    Handle OAuth callback from Google.

    After Google authentication, this endpoint receives the authorization code
    and exchanges it for session tokens. Creates a profile for new users.

    Args:
        code: Authorization code from OAuth provider
        error: Error message if OAuth failed

    Returns:
        AuthResponse with user_id, email, and session tokens
    """
    # Handle OAuth errors
    if error:
        raise HTTPException(
            status_code=400,
            detail=f"OAuth authentication failed: {error}"
        )

    if not code:
        raise HTTPException(
            status_code=400,
            detail="Missing authorization code"
        )

    try:
        # Exchange code for session
        auth_result = db.exchange_oauth_code(code)

        user_id = auth_result["user_id"]
        email = auth_result["email"]
        is_new_user = auth_result.get("is_new_user", False)

        # Create profile for new OAuth users
        if is_new_user:
            try:
                db.create_profile(
                    user_id=user_id,
                    email=email,
                    tos_accepted=True  # ToS acceptance implicit in OAuth signup
                )
            except Exception as profile_error:
                # Log but don't fail - user can still continue
                print(f"Warning: Failed to create profile for OAuth user {user_id}: {profile_error}")

        return AuthResponse(
            user_id=user_id,
            email=email,
            access_token=auth_result["access_token"],
            refresh_token=auth_result["refresh_token"]
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OAuth callback failed: {str(e)}"
        )


# --- Checkout Endpoints ---

@app.post("/checkout/create", response_model=CheckoutResponse)
async def create_checkout(
    request: CheckoutRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Create a Stripe Checkout session for a full video translation.

    Requires authentication. Creates a translation_job record with the price
    calculated from the video duration, then creates a Stripe Checkout session.

    Args:
        request: CheckoutRequest with preview_id
        current_user: Authenticated user from JWT token

    Returns:
        CheckoutResponse with checkout_url for redirect and translation_job_id

    Raises:
        400: Invalid preview_id
        403: Preview doesn't belong to user
        404: Preview not found
        500: Stripe or database error
    """
    preview_id = request.preview_id

    # Get the preview job
    preview_job = db.get_preview_job(preview_id)
    if not preview_job:
        raise HTTPException(
            status_code=404,
            detail="Preview job not found"
        )

    # Verify preview is completed
    if preview_job.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Preview is not ready. Status: {preview_job.get('status')}"
        )

    # Get video info to calculate price
    video_url = preview_job.get("video_url")
    if not video_url:
        raise HTTPException(
            status_code=400,
            detail="Preview job missing video URL"
        )

    # Extract video info for price calculation
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            if not info:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract video information"
                )
            duration = info.get('duration', 0)
            video_title = info.get('title', 'Video Translation')
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video URL: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get video info: {str(e)}"
        )

    # Calculate price
    price_cents = calculate_price_cents(duration)

    # Create translation job record
    try:
        translation_job = db.create_translation_job(
            user_id=current_user.user_id,
            preview_job_id=preview_id,
            price_cents=price_cents
        )
        translation_job_id = translation_job.get("id")
        if not translation_job_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to create translation job"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create translation job: {str(e)}"
        )

    # Create Stripe Checkout session
    try:
        checkout_result = db.create_stripe_checkout_session(
            translation_job_id=translation_job_id,
            price_cents=price_cents,
            video_title=video_title,
            customer_email=current_user.email
        )

        # Update translation job with Stripe session ID
        db.update_translation_job(
            translation_job_id,
            stripe_checkout_session_id=checkout_result["checkout_session_id"]
        )

        return CheckoutResponse(
            checkout_url=checkout_result["checkout_url"],
            translation_job_id=translation_job_id
        )

    except ValueError as e:
        # Cleanup: delete the translation job if Stripe fails
        # (optional - could leave for debugging)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create checkout session: {str(e)}"
        )


# --- Stripe Webhook Endpoint ---

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle Stripe webhook events.

    Verifies the webhook signature and processes events:
    - checkout.session.completed: Updates payment_status to paid and starts full translation

    Returns:
        200 OK on successful processing
        400 for invalid signatures or payloads
    """
    # Get the raw body for signature verification
    payload = await request.body()

    # Get the Stripe signature header
    signature = request.headers.get("stripe-signature")
    if not signature:
        raise HTTPException(
            status_code=400,
            detail="Missing Stripe-Signature header"
        )

    # Verify signature and construct event
    try:
        event = db.verify_stripe_webhook_signature(payload, signature)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # Handle the event
    event_type = event.get("type")

    if event_type == "checkout.session.completed":
        session = event.get("data", {}).get("object", {})
        checkout_session_id = session.get("id")

        if not checkout_session_id:
            return Response(status_code=200, content="No session ID in event")

        # Get the translation job by checkout session ID
        translation_job = db.get_translation_job_by_checkout_session(checkout_session_id)

        if not translation_job:
            # Job not found - could be a duplicate event or test webhook
            return Response(status_code=200, content="Translation job not found")

        translation_job_id = translation_job.get("id")
        preview_job_id = translation_job.get("preview_job_id")

        # Get Stripe payment intent ID from the session
        payment_intent_id = session.get("payment_intent")

        # Update translation job: mark payment as paid
        db.update_translation_job(
            translation_job_id,
            payment_status="paid",
            stripe_payment_intent_id=payment_intent_id,
            status="processing"
        )

        # Get preview job to extract video_url and target_language
        preview_job = db.get_preview_job(preview_job_id)

        if preview_job:
            video_url = preview_job.get("video_url")
            target_language = preview_job.get("target_language")

            if video_url and target_language:
                # Trigger full translation processing in background
                background_tasks.add_task(
                    process_full_translation,
                    translation_job_id,
                    video_url,
                    target_language
                )

    # Return 200 to acknowledge receipt
    return Response(status_code=200, content="Webhook received")


# --- Translation Job Endpoints ---

@app.get("/jobs/{job_id}", response_model=TranslationJobStatusResponse)
async def get_translation_job_status(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get the status of a paid translation job.

    Requires authentication. Only returns jobs owned by the current user.

    Args:
        job_id: UUID of the translation job
        current_user: Authenticated user from JWT token

    Returns:
        TranslationJobStatusResponse with status, progress, stage, and download_url when completed

    Raises:
        404: Job not found or not owned by user
    """
    # Get the translation job from database
    translation_job = db.get_translation_job(job_id)

    if not translation_job:
        raise HTTPException(status_code=404, detail="Translation job not found")

    # Verify ownership: job must belong to current user
    job_user_id = translation_job.get("user_id")
    if job_user_id != current_user.user_id:
        # Return 404 instead of 403 to not leak existence of other users' jobs
        raise HTTPException(status_code=404, detail="Translation job not found")

    # Build response
    status = translation_job.get("status", "pending")
    progress = translation_job.get("progress", 0)
    stage = translation_job.get("stage")
    error = translation_job.get("error_message")

    download_url = None

    # When completed, include download_url
    if status == "completed":
        output_file_path = translation_job.get("output_file_path")
        if output_file_path:
            try:
                # Generate signed URL with 24 hour expiration
                download_url = db.get_translation_signed_url(output_file_path, expires_in=86400)
            except Exception:
                # If signed URL generation fails, still return status
                pass

    return TranslationJobStatusResponse(
        job_id=job_id,
        status=status,
        progress=progress,
        stage=stage,
        error=error,
        download_url=download_url
    )


class TranslationDownloadResponse(BaseModel):
    download_url: str  # Signed URL for downloading the translated video
    expires_in: int  # Expiration time in seconds


class UserJobListItem(BaseModel):
    job_id: str  # UUID of the translation job
    video_title: Optional[str] = None  # Title of the translated video
    target_language: str  # Language code (e.g., "es", "ja", "fr")
    status: str  # "pending", "processing", "completed", "failed"
    created_at: str  # ISO format datetime
    download_available: bool  # Whether the download is ready


@app.get("/jobs", response_model=list[UserJobListItem])
async def list_user_jobs(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    List all translation jobs for the authenticated user.

    Returns a list of translation jobs ordered by created_at descending (most recent first).

    Args:
        current_user: Authenticated user from JWT token

    Returns:
        List of UserJobListItem with job_id, video_title, target_language, status, created_at, download_available
    """
    # Get all translation jobs for the user from database
    translation_jobs = db.list_translation_jobs_by_user(current_user.user_id)

    # Convert to response model
    result = []
    for job in translation_jobs:
        # Check if download is available (status is completed and has output file)
        status = job.get("status", "pending")
        output_file_path = job.get("output_file_path")
        download_available = status == "completed" and output_file_path is not None

        # Format created_at as ISO string
        created_at = job.get("created_at", "")
        if hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat()

        result.append(UserJobListItem(
            job_id=job.get("id", ""),
            video_title=job.get("video_title"),
            target_language=job.get("target_language", ""),
            status=status,
            created_at=created_at,
            download_available=download_available
        ))

    return result


@app.get("/jobs/{job_id}/download", response_model=TranslationDownloadResponse)
async def download_translation_job(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get a signed download URL for a completed translation job.

    Requires authentication. Only returns download URL for jobs owned by the
    current user that have completed processing.

    Args:
        job_id: UUID of the translation job
        current_user: Authenticated user from JWT token

    Returns:
        TranslationDownloadResponse with download_url (signed URL, expires in 24 hours)

    Raises:
        401: Missing or invalid authentication token
        404: Job not found, not owned by user, or not completed
    """
    # Get the translation job from database
    translation_job = db.get_translation_job(job_id)

    if not translation_job:
        raise HTTPException(status_code=404, detail="Translation job not found")

    # Verify ownership: job must belong to current user
    job_user_id = translation_job.get("user_id")
    if job_user_id != current_user.user_id:
        # Return 404 instead of 403 to not leak existence of other users' jobs
        raise HTTPException(status_code=404, detail="Translation job not found")

    # Verify job is completed
    status = translation_job.get("status")
    if status != "completed":
        raise HTTPException(
            status_code=404,
            detail=f"Translation not ready for download. Status: {status}"
        )

    # Get the output file path
    output_file_path = translation_job.get("output_file_path")
    if not output_file_path:
        raise HTTPException(
            status_code=404,
            detail="Translation output file not found"
        )

    # Generate signed URL with 24 hour expiration (86400 seconds)
    expires_in = 86400
    try:
        download_url = db.get_translation_signed_url(output_file_path, expires_in=expires_in)
        if not download_url:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate download URL"
            )
        return TranslationDownloadResponse(
            download_url=download_url,
            expires_in=expires_in
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid storage path: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate download URL: {str(e)}"
        )


async def process_full_translation(translation_job_id: str, video_url: str, target_language: str):
    """
    Background task to process a full video translation after payment.

    Processes the entire video duration (no 60-second limit).
    Pipeline: download -> separate -> diarize -> transcribe -> translate -> TTS -> mix -> subtitles -> merge -> upload

    Note: Could optimize by reusing preview transcription for first 60 seconds,
    but for simplicity we re-process the entire video to ensure quality.
    """
    try:
        # Initialize job tracking in memory for helper functions
        jobs[translation_job_id] = {
            "job_id": translation_job_id,
            "status": "processing",
            "progress": 0,
            "stage": "initializing",
        }

        # Update status in database
        db.update_translation_job(
            translation_job_id,
            status="processing",
            progress=0,
            stage="initializing"
        )

        # 1. Download full video (no duration limit)
        db.update_translation_job(translation_job_id, stage="download", progress=5)
        video_path, audio_path, info = await download_video(
            video_url, translation_job_id, duration_limit=None
        )
        db.update_translation_job(translation_job_id, progress=25)

        # 2. Separate audio into vocals and background
        db.update_translation_job(translation_job_id, stage="separate", progress=26)
        vocals_path, background_path = await separate_audio(audio_path, translation_job_id)
        db.update_translation_job(translation_job_id, progress=30)

        # 3. Run speaker diarization on vocals
        db.update_translation_job(translation_job_id, stage="diarize", progress=31)
        diarization_segments, speaker_voice_urls = await diarize_speakers(vocals_path, translation_job_id)
        db.update_translation_job(translation_job_id, progress=40)

        # 4. Transcribe vocals (cleaner input without background noise)
        db.update_translation_job(translation_job_id, stage="transcribe", progress=41)
        segments = await transcribe_audio(vocals_path, translation_job_id)
        db.update_translation_job(translation_job_id, progress=48)

        # 5. Translate segments (preserves timestamps)
        db.update_translation_job(translation_job_id, stage="translate", progress=50)
        translated_segments = await translate_segments(segments, target_language, translation_job_id)
        db.update_translation_job(translation_job_id, progress=55)

        # 6. Synthesize speech (TTS) - use multi-speaker if diarization available
        db.update_translation_job(translation_job_id, stage="synthesize", progress=60)
        if speaker_voice_urls and diarization_segments:
            # Multi-speaker synthesis using per-segment voice matching
            new_audio = await synthesize_segments_multi_speaker(
                translated_segments=translated_segments,
                diarization_segments=diarization_segments,
                speaker_voice_urls=speaker_voice_urls,
                target_lang=target_language,
                job_id=translation_job_id
            )
        else:
            # Fallback: single speaker synthesis (combine all text)
            translated_text = " ".join(
                seg["translated_text"] for seg in translated_segments if seg.get("translated_text")
            )
            new_audio = await synthesize_speech(translated_text, vocals_path, target_language, translation_job_id)
        db.update_translation_job(translation_job_id, progress=82)

        # 7. Mix translated speech with original background audio (30% volume)
        db.update_translation_job(translation_job_id, stage="mix", progress=83)
        # Only mix if background_path is different from audio_path (separation succeeded)
        if background_path != audio_path:
            mixed_audio = mix_audio_with_background(new_audio, background_path)
        else:
            mixed_audio = new_audio
        db.update_translation_job(translation_job_id, progress=85)

        # 8. Generate SRT subtitles from translated segments
        db.update_translation_job(translation_job_id, stage="subtitles", progress=86)
        subtitle_path = generate_srt(translated_segments, job_id=translation_job_id)
        subtitle_lang = get_iso_639_2_code(target_language)
        db.update_translation_job(translation_job_id, progress=88)

        # 9. Merge translated audio with video (with embedded subtitles)
        db.update_translation_job(translation_job_id, stage="merge", progress=90)
        output_path = await merge_audio_video(
            video_path, mixed_audio, translation_job_id, info["title"],
            subtitle_path=subtitle_path, subtitle_lang=subtitle_lang
        )
        db.update_translation_job(translation_job_id, progress=95)

        # 10. Upload to Supabase Storage
        db.update_translation_job(translation_job_id, stage="upload", progress=95)
        storage_path = db.upload_translation_to_storage(translation_job_id, str(output_path))

        # 11. Update job as completed with file path
        db.update_translation_job(
            translation_job_id,
            status="completed",
            progress=100,
            stage="done",
            output_file_path=storage_path
        )

        # Update in-memory job
        update_job(
            translation_job_id,
            status="completed",
            progress=100,
            stage="done",
            output_file=str(output_path)
        )

    except Exception as e:
        # Update both database and in-memory state
        db.update_translation_job(
            translation_job_id,
            status="failed",
            error_message=str(e),
            stage="error"
        )
        if translation_job_id in jobs:
            update_job(
                translation_job_id,
                status="failed",
                error=str(e),
                stage="error"
            )


@app.post("/translate", response_model=JobStatus)
async def start_translation(request: TranslateRequest, background_tasks: BackgroundTasks):
    """
    Start a video translation job.

    Returns a job_id that can be used to check status.
    """
    # Validate language
    if request.target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {request.target_language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    # Check API keys
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    if not REPLICATE_API_TOKEN:
        raise HTTPException(status_code=500, detail="Replicate API token not configured")

    # Create job
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "stage": "queued",
        "video_url": request.video_url,
        "target_language": request.target_language,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    # Start background processing
    background_tasks.add_task(
        process_translation,
        job_id,
        request.video_url,
        request.target_language
    )

    return JobStatus(
        job_id=job_id,
        status="pending",
        progress=0,
        stage="queued"
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a translation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        stage=job.get("stage"),
        error=job.get("error"),
        output_url=job.get("output_url")
    )


@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download the translated video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )

    output_file = job.get("output_file")
    if not output_file or not Path(output_file).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        output_file,
        media_type="video/mp4",
        filename=Path(output_file).name
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete files
    job_dir = OUTPUT_DIR / job_id
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir)

    # Remove from jobs dict
    del jobs[job_id]

    return {"status": "deleted", "job_id": job_id}


# --- Development server ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
