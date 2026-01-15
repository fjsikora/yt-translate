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
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import httpx
import replicate
import yt_dlp
from deep_translator import GoogleTranslator
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

import db

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/yt-translate"))
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "600"))  # 10 minutes default
PRICE_PER_MINUTE_CENTS = int(os.getenv("PRICE_PER_MINUTE_CENTS", "50"))  # $0.50 per minute default
PREVIEW_DURATION_SECONDS = int(os.getenv("PREVIEW_DURATION_SECONDS", "60"))  # Free preview duration

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


async def transcribe_audio(audio_path: Path, job_id: str) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    update_job(job_id, stage="transcribe", progress=30)

    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

    update_job(job_id, progress=45)
    return transcript


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


async def synthesize_speech(text: str, audio_prompt_path: Path, target_lang: str, job_id: str) -> Path:
    """Generate speech using Replicate Chatterbox API."""
    update_job(job_id, stage="synthesize", progress=60)

    job_dir = OUTPUT_DIR / job_id
    output_audio = job_dir / "translated_audio.wav"

    # Upload audio prompt to a temporary URL (Replicate needs a URL)
    # For production, use cloud storage (S3, GCS, etc.)
    # For now, we'll use the file directly if Replicate supports it

    # Read the first 10 seconds of audio for voice cloning reference
    voice_sample = job_dir / "voice_sample.wav"
    subprocess.run([
        'ffmpeg', '-i', str(audio_prompt_path),
        '-t', '10', '-y', str(voice_sample)
    ], capture_output=True, check=True)

    update_job(job_id, progress=65)

    # Call Replicate API
    # Note: Replicate needs a URL for the audio file, so we'll need to handle this
    # For local testing, you may need to use a file hosting service

    try:
        output = replicate.run(
            "resemble-ai/chatterbox-multilingual",
            input={
                "text": text,
                "language_id": target_lang,
                # Note: audio_prompt needs to be a URL in production
                # For local dev, you might need to upload to a temp file service
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
            # Output might be file-like
            with open(output_audio, "wb") as f:
                for chunk in output:
                    f.write(chunk)

    except Exception as e:
        raise RuntimeError(f"TTS synthesis failed: {e}")

    update_job(job_id, progress=85)
    return output_audio


async def merge_audio_video(video_path: Path, audio_path: Path, job_id: str, title: str) -> Path:
    """Merge translated audio with original video."""
    update_job(job_id, stage="merge", progress=90)

    job_dir = OUTPUT_DIR / job_id
    safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:40]
    output_path = job_dir / f"{safe_title}_translated.mp4"

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
    """Main translation pipeline."""
    try:
        update_job(job_id, status="processing", progress=0)

        # 1. Download video
        video_path, audio_path, info = await download_video(video_url, job_id)

        # 2. Transcribe
        transcript = await transcribe_audio(audio_path, job_id)

        # 3. Translate
        translated = await translate_text(transcript, target_language, job_id)

        # 4. Synthesize (TTS)
        new_audio = await synthesize_speech(translated, audio_path, target_language, job_id)

        # 5. Merge
        output = await merge_audio_video(video_path, new_audio, job_id, info["title"])

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


# --- API Endpoints ---

@app.get("/")
async def root():
    """Redirect to frontend."""
    return RedirectResponse(url="/static/index.html")


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
    Pipeline: download -> transcribe -> translate -> TTS -> merge -> upload
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

        # 2. Transcribe audio
        db.update_preview_job(preview_id, stage="transcribe", progress=30)
        transcript = await transcribe_audio(audio_path, preview_id)
        db.update_preview_job(preview_id, progress=45)

        # 3. Translate text
        db.update_preview_job(preview_id, stage="translate", progress=50)
        translated = await translate_text(transcript, target_language, preview_id)
        db.update_preview_job(preview_id, progress=55)

        # 4. Synthesize speech (TTS with voice cloning)
        db.update_preview_job(preview_id, stage="synthesize", progress=60)
        new_audio = await synthesize_speech(translated, audio_path, target_language, preview_id)
        db.update_preview_job(preview_id, progress=85)

        # 5. Merge translated audio with video
        db.update_preview_job(preview_id, stage="merge", progress=90)
        output_path = await merge_audio_video(video_path, new_audio, preview_id, info["title"])
        db.update_preview_job(preview_id, progress=95)

        # 6. Upload to Supabase Storage
        db.update_preview_job(preview_id, stage="upload", progress=95)
        storage_path = db.upload_preview_to_storage(preview_id, str(output_path))

        # 7. Update job as completed with file path
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
async def create_preview(request: PreviewRequest, background_tasks: BackgroundTasks):
    """
    Start a free preview translation job (first 60 seconds only).

    Creates a preview job record in the database and starts background processing.
    Guest users are identified by session_id.
    """
    # Validate language
    if request.target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {request.target_language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    # Validate session_id is not empty
    if not request.session_id or not request.session_id.strip():
        raise HTTPException(
            status_code=400,
            detail="session_id is required"
        )

    # Validate video URL by attempting to extract info
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(request.video_url, download=False)
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
            session_id=request.session_id.strip(),
            video_url=request.video_url,
            target_language=request.target_language
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
        request.video_url,
        request.target_language
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
