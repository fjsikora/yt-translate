"""
RunPod Serverless API Client.

This module provides async functions for interacting with the RunPod serverless
endpoint that runs the video translation worker.

Environment variables:
- RUNPOD_API_KEY: API key for authenticating with RunPod
- RUNPOD_ENDPOINT_ID: The serverless endpoint ID (e.g., "abc123xyz")
"""

import os
from typing import Any, Optional
from typing_extensions import TypedDict
import httpx


# RunPod API configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_BASE_URL = "https://api.runpod.ai/v2"

# Timeout configuration
RUNPOD_TIMEOUT_SECONDS = 30.0  # Timeout for API requests


class RunPodError(Exception):
    """Exception raised for RunPod API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class JobInput(TypedDict, total=False):
    """Input parameters for a translation job."""
    video_url: str
    target_lang: str
    voice_sample_url: Optional[str]


class JobOutput(TypedDict, total=False):
    """Output from a completed translation job."""
    status: str
    output_url: str
    duration: float
    segments_count: int
    speakers_count: int
    error: Optional[str]


class JobStatus(TypedDict):
    """Status of a RunPod job."""
    id: str
    status: str  # "IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED"
    output: Optional[JobOutput]
    error: Optional[str]


def _get_headers() -> dict[str, str]:
    """Get authorization headers for RunPod API requests."""
    if not RUNPOD_API_KEY:
        raise RunPodError("RUNPOD_API_KEY environment variable not set")
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


def _get_endpoint_url(path: str = "") -> str:
    """Construct the full endpoint URL."""
    if not RUNPOD_ENDPOINT_ID:
        raise RunPodError("RUNPOD_ENDPOINT_ID environment variable not set")
    return f"{RUNPOD_API_BASE_URL}/{RUNPOD_ENDPOINT_ID}{path}"


async def submit_job(
    video_url: str,
    target_lang: str,
    voice_sample_url: Optional[str] = None,
) -> str:
    """
    Submit a new translation job to RunPod serverless.

    Args:
        video_url: URL of the video to translate (YouTube, direct URL, etc.)
        target_lang: Target language code (e.g., "es", "ja", "fr")
        voice_sample_url: Optional URL to a custom voice sample for cloning

    Returns:
        job_id: The unique identifier for the submitted job

    Raises:
        RunPodError: If the API request fails or returns an error
    """
    job_input: JobInput = {
        "video_url": video_url,
        "target_lang": target_lang,
    }
    if voice_sample_url:
        job_input["voice_sample_url"] = voice_sample_url

    payload = {"input": job_input}

    async with httpx.AsyncClient(timeout=RUNPOD_TIMEOUT_SECONDS) as client:
        try:
            response = await client.post(
                _get_endpoint_url("/run"),
                headers=_get_headers(),
                json=payload,
            )
        except httpx.TimeoutException as e:
            raise RunPodError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            raise RunPodError(f"Request failed: {e}") from e

        if response.status_code != 200:
            _handle_error_response(response)

        data = response.json()
        job_id = data.get("id")
        if not job_id:
            raise RunPodError("Response missing job ID", details=data)

        return job_id


async def get_job_status(job_id: str) -> JobStatus:
    """
    Get the current status of a translation job.

    Args:
        job_id: The unique identifier of the job

    Returns:
        JobStatus dict containing:
            - id: Job ID
            - status: One of "IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED"
            - output: Job output if completed (contains output_url, duration, etc.)
            - error: Error message if failed

    Raises:
        RunPodError: If the API request fails or returns an error
    """
    async with httpx.AsyncClient(timeout=RUNPOD_TIMEOUT_SECONDS) as client:
        try:
            response = await client.get(
                _get_endpoint_url(f"/status/{job_id}"),
                headers=_get_headers(),
            )
        except httpx.TimeoutException as e:
            raise RunPodError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            raise RunPodError(f"Request failed: {e}") from e

        if response.status_code != 200:
            _handle_error_response(response)

        data = response.json()

        return JobStatus(
            id=data.get("id", job_id),
            status=data.get("status", "UNKNOWN"),
            output=data.get("output"),
            error=data.get("error"),
        )


async def cancel_job(job_id: str) -> bool:
    """
    Cancel a running or queued job.

    Args:
        job_id: The unique identifier of the job to cancel

    Returns:
        True if the job was successfully cancelled

    Raises:
        RunPodError: If the API request fails or returns an error
    """
    async with httpx.AsyncClient(timeout=RUNPOD_TIMEOUT_SECONDS) as client:
        try:
            response = await client.post(
                _get_endpoint_url(f"/cancel/{job_id}"),
                headers=_get_headers(),
            )
        except httpx.TimeoutException as e:
            raise RunPodError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            raise RunPodError(f"Request failed: {e}") from e

        if response.status_code != 200:
            _handle_error_response(response)

        data = response.json()
        # RunPod returns status after cancel attempt
        return data.get("status") in ("CANCELLED", "CANCELLING")


def _handle_error_response(response: httpx.Response) -> None:
    """
    Handle error responses from the RunPod API.

    Args:
        response: The HTTP response object

    Raises:
        RunPodError: Always raises with appropriate error details
    """
    try:
        error_data = response.json()
    except Exception:
        error_data = {"raw": response.text}

    status_code = response.status_code

    # Common error codes
    if status_code == 401:
        raise RunPodError(
            "Invalid or missing API key",
            status_code=status_code,
            details=error_data,
        )
    elif status_code == 404:
        raise RunPodError(
            "Endpoint or job not found",
            status_code=status_code,
            details=error_data,
        )
    elif status_code == 429:
        raise RunPodError(
            "Rate limit exceeded",
            status_code=status_code,
            details=error_data,
        )
    elif status_code >= 500:
        raise RunPodError(
            f"RunPod server error (HTTP {status_code})",
            status_code=status_code,
            details=error_data,
        )
    else:
        raise RunPodError(
            f"API error (HTTP {status_code}): {error_data.get('error', 'Unknown error')}",
            status_code=status_code,
            details=error_data,
        )


def map_runpod_status_to_api(runpod_status: str) -> tuple[str, Optional[str]]:
    """
    Map RunPod job status to the existing API response format.

    Args:
        runpod_status: RunPod status string

    Returns:
        Tuple of (status, stage) matching the existing API format:
            - status: "pending", "processing", "completed", "failed"
            - stage: Human-readable stage description or None
    """
    status_map: dict[str, tuple[str, Optional[str]]] = {
        "IN_QUEUE": ("pending", "queued"),
        "IN_PROGRESS": ("processing", "translating"),
        "COMPLETED": ("completed", None),
        "FAILED": ("failed", None),
        "CANCELLED": ("failed", "cancelled"),
        "CANCELLING": ("processing", "cancelling"),
    }
    return status_map.get(runpod_status, ("pending", None))
