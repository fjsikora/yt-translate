"""
Railway API Gateway for Video Translation.

This module provides the FastAPI server that handles API requests for video
translation, delegating the actual processing to RunPod serverless workers.
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from languages import SUPPORTED_LANGUAGES
from runpod_client import (
    RunPodError,
    submit_job,
    get_job_status as runpod_get_job_status,
    cancel_job,
    map_runpod_status_to_api,
)


# --- Pydantic Models ---

class TranslateRequest(BaseModel):
    """Request body for starting a translation job."""
    video_url: str  # Supports YouTube, Rumble, Vimeo, and 1700+ sites via yt-dlp
    target_language: str  # Language code (e.g., "es", "ja", "fr")
    voice_sample_url: Optional[str] = None  # Optional custom voice sample URL


class JobStatus(BaseModel):
    """Status response for a translation job."""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int  # 0-100
    stage: Optional[str] = None
    error: Optional[str] = None
    output_url: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


# --- FastAPI App Setup ---

app = FastAPI(
    title="Video Translation API",
    description="API for translating videos using RunPod serverless workers",
    version="2.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- API Endpoints ---

@app.get("/")
async def root():
    """Redirect to frontend."""
    return RedirectResponse(url="/static/index.html")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Railway/load balancers."""
    return HealthResponse(status="ok", version="2.0.0")


@app.get("/languages")
async def list_languages():
    """List supported target languages."""
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/translate", response_model=JobStatus)
async def start_translation(request: TranslateRequest):
    """
    Start a video translation job.

    Submits the job to RunPod serverless and returns a job_id
    that can be used to check status via /status/{job_id}.
    """
    # Validate language
    if request.target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {request.target_language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    try:
        # Submit job to RunPod
        job_id = await submit_job(
            video_url=request.video_url,
            target_lang=request.target_language,
            voice_sample_url=request.voice_sample_url,
        )

        return JobStatus(
            job_id=job_id,
            status="pending",
            progress=0,
            stage="queued",
        )

    except RunPodError as e:
        # Map RunPod errors to appropriate HTTP errors
        if e.status_code == 401:
            raise HTTPException(status_code=500, detail="RunPod API authentication failed")
        elif e.status_code == 429:
            raise HTTPException(status_code=503, detail="Service temporarily unavailable (rate limited)")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to submit job: {e.message}")


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Get the status of a translation job.

    Polls RunPod for the current job status and returns it in the
    standard API format for backwards compatibility.
    """
    try:
        # Get status from RunPod
        runpod_status = await runpod_get_job_status(job_id)

        # Map RunPod status to API status
        status, stage = map_runpod_status_to_api(runpod_status["status"])

        # Calculate progress based on status
        if status == "completed":
            progress = 100
        elif status == "processing":
            progress = 50  # RunPod doesn't provide detailed progress
        elif status == "failed":
            progress = 0
        else:
            progress = 0

        # Extract output_url from completed job output
        output_url = None
        error = runpod_status.get("error")

        if runpod_status.get("output"):
            output = runpod_status["output"]
            output_url = output.get("output_url")
            # Check for error in output
            if output.get("error"):
                error = output.get("error")

        return JobStatus(
            job_id=job_id,
            status=status,
            progress=progress,
            stage=stage,
            error=error,
            output_url=output_url,
        )

    except RunPodError as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail="Job not found")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to get job status: {e.message}")


@app.post("/cancel/{job_id}")
async def cancel_translation(job_id: str):
    """
    Cancel a running or queued translation job.

    Attempts to cancel the job on RunPod. Returns success status.
    """
    try:
        success = await cancel_job(job_id)
        return {"success": success, "job_id": job_id}

    except RunPodError as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail="Job not found")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e.message}")
