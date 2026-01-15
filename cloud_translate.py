#!/usr/bin/env python3
"""
Universal Video Translator - Cloud API Version

A FastAPI service that translates videos from 1700+ sites using cloud APIs:
- yt-dlp: Download video & extract audio (local)
- OpenAI Whisper API: Speech-to-text transcription
- Google Translate: Text translation
- Replicate API: Chatterbox TTS with voice cloning
- ffmpeg: Audio/video merging (local)

Designed for deployment on Self-hosted, Render, or similar platforms.
No GPU required - all heavy lifting done by cloud APIs.
"""

import asyncio
import os
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

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/yt-translate"))
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "600"))  # 10 minutes default
PROCESSING_RATE = int(os.getenv("PROCESSING_RATE", "50"))  # $0.50 per minute default

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


async def download_video(url: str, job_id: str) -> tuple[Path, Path, dict]:
    """Download video from any supported site and extract audio."""
    update_job(job_id, stage="download", progress=5)

    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Get video info first
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info['id']
        title = info.get('title', video_id)
        duration = info.get('duration', 0)

        if duration > MAX_VIDEO_DURATION:
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

    # Extract audio
    subprocess.run([
        'ffmpeg', '-i', str(video_path),
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        '-y', str(audio_path)
    ], capture_output=True, check=True)

    update_job(job_id, progress=25)

    return video_path, audio_path, {"title": title, "duration": duration, "video_id": video_id}


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


def calculate_processing_cost(duration_seconds: int) -> int:
    """
    Calculate price in cents based on video duration.

    Price is calculated per minute, rounded up.
    Minimum price is 1 minute.
    """
    minutes = max(1, (duration_seconds + 59) // 60)  # Round up to nearest minute
    return minutes * PROCESSING_RATE


# --- API Endpoints ---

@app.get("/")
async def root():
    """Redirect to frontend."""
    return RedirectResponse(url="/static/index.html")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Self-hosted/load balancers."""
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

            processing_cost = calculate_processing_cost(duration)

            return VideoInfoResponse(
                video_title=title,
                duration_seconds=duration,
                thumbnail_url=thumbnail,
                price_quote=processing_cost
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
