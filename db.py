"""
Database module for Supabase integration.

Provides client initialization and helper functions for CRUD operations
on profiles, preview_jobs, and translation_jobs tables.
"""

import os
from typing import Optional
from datetime import datetime

from supabase import create_client, Client


# Environment variables for Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Global client instance (lazy initialization)
_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """
    Get or create Supabase client instance.

    Uses service key for backend operations (bypasses RLS).

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_SERVICE_KEY not set.
    """
    global _supabase_client

    if _supabase_client is None:
        if not SUPABASE_URL:
            raise ValueError("SUPABASE_URL environment variable not set")
        if not SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_SERVICE_KEY environment variable not set")

        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    return _supabase_client


# --- Profile CRUD Operations ---

def create_profile(user_id: str, email: str, tos_accepted: bool = False) -> dict:
    """
    Create a new user profile.

    Args:
        user_id: UUID from Supabase Auth
        email: User's email address
        tos_accepted: Whether user accepted Terms of Service

    Returns:
        Created profile record
    """
    client = get_supabase_client()

    data = {
        "id": user_id,
        "email": email,
    }

    if tos_accepted:
        data["tos_accepted_at"] = datetime.utcnow().isoformat()

    result = client.table("profiles").insert(data).execute()
    return result.data[0] if result.data else {}


def get_profile(user_id: str) -> Optional[dict]:
    """
    Get a user profile by ID.

    Args:
        user_id: UUID of the user

    Returns:
        Profile record or None if not found
    """
    client = get_supabase_client()

    result = client.table("profiles").select("*").eq("id", user_id).execute()
    return result.data[0] if result.data else None


def update_profile(user_id: str, **kwargs) -> Optional[dict]:
    """
    Update a user profile.

    Args:
        user_id: UUID of the user
        **kwargs: Fields to update (email, tos_accepted_at)

    Returns:
        Updated profile record or None
    """
    client = get_supabase_client()

    result = client.table("profiles").update(kwargs).eq("id", user_id).execute()
    return result.data[0] if result.data else None


# --- Preview Jobs CRUD Operations ---

def create_preview_job(
    session_id: str,
    video_url: str,
    target_language: str,
    user_id: Optional[str] = None
) -> dict:
    """
    Create a new preview job.

    Args:
        session_id: Anonymous session identifier
        video_url: URL of the video to preview
        target_language: Target language code
        user_id: Optional user ID if logged in

    Returns:
        Created preview job record
    """
    client = get_supabase_client()

    data = {
        "session_id": session_id,
        "video_url": video_url,
        "target_language": target_language,
        "status": "pending",
    }

    if user_id:
        data["user_id"] = user_id

    result = client.table("preview_jobs").insert(data).execute()
    return result.data[0] if result.data else {}


def get_preview_job(preview_id: str) -> Optional[dict]:
    """
    Get a preview job by ID.

    Args:
        preview_id: UUID of the preview job

    Returns:
        Preview job record or None if not found
    """
    client = get_supabase_client()

    result = client.table("preview_jobs").select("*").eq("id", preview_id).execute()
    return result.data[0] if result.data else None


def get_preview_job_by_session(session_id: str, preview_id: str) -> Optional[dict]:
    """
    Get a preview job by session ID and preview ID.

    Used for guest access validation.

    Args:
        session_id: Anonymous session identifier
        preview_id: UUID of the preview job

    Returns:
        Preview job record or None if not found
    """
    client = get_supabase_client()

    result = (
        client.table("preview_jobs")
        .select("*")
        .eq("id", preview_id)
        .eq("session_id", session_id)
        .execute()
    )
    return result.data[0] if result.data else None


def update_preview_job(preview_id: str, **kwargs) -> Optional[dict]:
    """
    Update a preview job.

    Args:
        preview_id: UUID of the preview job
        **kwargs: Fields to update (status, progress, stage, preview_file_path, error_message)

    Returns:
        Updated preview job record or None
    """
    client = get_supabase_client()

    result = client.table("preview_jobs").update(kwargs).eq("id", preview_id).execute()
    return result.data[0] if result.data else None


def list_preview_jobs_by_user(user_id: str) -> list[dict]:
    """
    List all preview jobs for a user.

    Args:
        user_id: UUID of the user

    Returns:
        List of preview job records
    """
    client = get_supabase_client()

    result = (
        client.table("preview_jobs")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


# --- Translation Jobs CRUD Operations ---

def create_translation_job(
    user_id: str,
    preview_job_id: str,
    price_cents: int
) -> dict:
    """
    Create a new translation job (for paid full translations).

    Args:
        user_id: UUID of the paying user
        preview_job_id: UUID of the associated preview job
        price_cents: Price in cents

    Returns:
        Created translation job record
    """
    client = get_supabase_client()

    data = {
        "user_id": user_id,
        "preview_job_id": preview_job_id,
        "price_cents": price_cents,
        "payment_status": "pending",
        "status": "pending",
    }

    result = client.table("translation_jobs").insert(data).execute()
    return result.data[0] if result.data else {}


def get_translation_job(job_id: str) -> Optional[dict]:
    """
    Get a translation job by ID.

    Args:
        job_id: UUID of the translation job

    Returns:
        Translation job record or None if not found
    """
    client = get_supabase_client()

    result = client.table("translation_jobs").select("*").eq("id", job_id).execute()
    return result.data[0] if result.data else None


def get_translation_job_by_checkout_session(checkout_session_id: str) -> Optional[dict]:
    """
    Get a translation job by Stripe checkout session ID.

    Args:
        checkout_session_id: Stripe checkout session ID

    Returns:
        Translation job record or None if not found
    """
    client = get_supabase_client()

    result = (
        client.table("translation_jobs")
        .select("*")
        .eq("stripe_checkout_session_id", checkout_session_id)
        .execute()
    )
    return result.data[0] if result.data else None


def update_translation_job(job_id: str, **kwargs) -> Optional[dict]:
    """
    Update a translation job.

    Args:
        job_id: UUID of the translation job
        **kwargs: Fields to update (payment_status, status, stripe fields, output_file_path, etc.)

    Returns:
        Updated translation job record or None
    """
    client = get_supabase_client()

    result = client.table("translation_jobs").update(kwargs).eq("id", job_id).execute()
    return result.data[0] if result.data else None


def list_translation_jobs_by_user(user_id: str) -> list[dict]:
    """
    List all translation jobs for a user.

    Args:
        user_id: UUID of the user

    Returns:
        List of translation job records
    """
    client = get_supabase_client()

    result = (
        client.table("translation_jobs")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


# --- Storage Operations ---

PREVIEW_BUCKET = "previews"


def upload_preview_to_storage(preview_id: str, file_path: str) -> str:
    """
    Upload a preview video file to Supabase Storage.

    Args:
        preview_id: UUID of the preview job (used in storage path)
        file_path: Local path to the video file to upload

    Returns:
        Storage path of the uploaded file (e.g., "previews/{preview_id}/preview.mp4")

    Raises:
        Exception: If upload fails
    """
    client = get_supabase_client()

    # Storage path: previews/{preview_id}/preview.mp4
    storage_path = f"{preview_id}/preview.mp4"

    # Read file and upload
    with open(file_path, "rb") as f:
        file_content = f.read()

    # Upload to Supabase Storage
    result = client.storage.from_(PREVIEW_BUCKET).upload(
        path=storage_path,
        file=file_content,
        file_options={"content-type": "video/mp4"}
    )

    # Return the full storage path
    return f"{PREVIEW_BUCKET}/{storage_path}"


def get_preview_signed_url(storage_path: str, expires_in: int = 3600) -> str:
    """
    Get a signed URL for accessing a preview file.

    Args:
        storage_path: Full storage path (e.g., "previews/{preview_id}/preview.mp4")
        expires_in: URL expiration time in seconds (default: 1 hour)

    Returns:
        Signed URL for accessing the file
    """
    client = get_supabase_client()

    # Extract bucket and path
    parts = storage_path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid storage path: {storage_path}")

    bucket = parts[0]
    path = parts[1]

    result = client.storage.from_(bucket).create_signed_url(path, expires_in)
    return result.get("signedURL", "")


# --- Connection Test ---

def test_connection() -> bool:
    """
    Test the Supabase connection.

    Returns:
        True if connection successful, raises exception otherwise.
    """
    client = get_supabase_client()

    # Try to query profiles table (should work even if empty)
    client.table("profiles").select("id").limit(1).execute()

    return True
