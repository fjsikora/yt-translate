"""
Database module for Supabase integration and Stripe payments.

Provides client initialization and helper functions for CRUD operations
on profiles, preview_jobs, and translation_jobs tables.
Also includes Stripe Checkout session creation.
"""

import os
from typing import Optional
from datetime import datetime

import stripe
from supabase import create_client, Client


# Environment variables for Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Environment variables for Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
STRIPE_SUCCESS_URL = os.getenv("STRIPE_SUCCESS_URL", "http://localhost:8000/success")
STRIPE_CANCEL_URL = os.getenv("STRIPE_CANCEL_URL", "http://localhost:8000/")

# Configure Stripe
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

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
        "stage": "queued",
        "progress": 0,
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
TRANSLATIONS_BUCKET = "translations"
VOICE_SAMPLES_BUCKET = "voice-samples"
TEMP_BUCKET = "tmp-files"


def upload_to_storage(
    bucket: str,
    job_id: str,
    file_path: str,
    filename: str,
    content_type: str
) -> str:
    """
    Upload a file to Supabase Storage.

    This is the generic upload function used by all storage operations.

    Args:
        bucket: Storage bucket name (e.g., "previews", "translations", "voice-samples")
        job_id: UUID of the job (used as folder in storage path)
        file_path: Local path to the file to upload
        filename: Name for the file in storage (e.g., "preview.mp4", "speaker1.wav")
        content_type: MIME type of the file (e.g., "video/mp4", "audio/wav")

    Returns:
        Full storage path of the uploaded file (e.g., "bucket/{job_id}/{filename}")

    Raises:
        Exception: If upload fails
    """
    client = get_supabase_client()

    # Storage path: {job_id}/{filename}
    storage_path = f"{job_id}/{filename}"

    # Read file and upload
    with open(file_path, "rb") as f:
        file_content = f.read()

    # Upload to Supabase Storage (upsert to overwrite existing files)
    client.storage.from_(bucket).upload(
        path=storage_path,
        file=file_content,
        file_options={"content-type": content_type, "upsert": "true"}
    )

    # Return the full storage path
    return f"{bucket}/{storage_path}"


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
    return upload_to_storage(
        bucket=PREVIEW_BUCKET,
        job_id=preview_id,
        file_path=file_path,
        filename="preview.mp4",
        content_type="video/mp4"
    )


def get_signed_url(bucket: str, storage_path: str, expires_in: int = 3600) -> str:
    """
    Get a signed URL for accessing a file in Supabase Storage.

    This is the generic signed URL function used by all storage operations.

    Args:
        bucket: Storage bucket name (e.g., "previews", "translations", "voice-samples")
        storage_path: Path within the bucket (e.g., "{job_id}/preview.mp4")
        expires_in: URL expiration time in seconds (default: 1 hour)

    Returns:
        Signed URL for accessing the file
    """
    client = get_supabase_client()

    result = client.storage.from_(bucket).create_signed_url(storage_path, expires_in)
    return result.get("signedURL", "")


def get_preview_signed_url(storage_path: str, expires_in: int = 3600) -> str:
    """
    Get a signed URL for accessing a preview file.

    Args:
        storage_path: Full storage path (e.g., "previews/{preview_id}/preview.mp4")
        expires_in: URL expiration time in seconds (default: 1 hour)

    Returns:
        Signed URL for accessing the file
    """
    # Extract bucket and path from full storage path
    parts = storage_path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid storage path: {storage_path}")

    bucket = parts[0]
    path = parts[1]

    return get_signed_url(bucket, path, expires_in)


def upload_translation_to_storage(translation_job_id: str, file_path: str) -> str:
    """
    Upload a full translation video file to Supabase Storage.

    Args:
        translation_job_id: UUID of the translation job (used in storage path)
        file_path: Local path to the video file to upload

    Returns:
        Storage path of the uploaded file (e.g., "translations/{job_id}/translation.mp4")

    Raises:
        Exception: If upload fails
    """
    return upload_to_storage(
        bucket=TRANSLATIONS_BUCKET,
        job_id=translation_job_id,
        file_path=file_path,
        filename="translation.mp4",
        content_type="video/mp4"
    )


def get_translation_signed_url(storage_path: str, expires_in: int = 86400) -> str:
    """
    Get a signed URL for accessing a translation file.

    Args:
        storage_path: Full storage path (e.g., "translations/{job_id}/translation.mp4")
        expires_in: URL expiration time in seconds (default: 24 hours)

    Returns:
        Signed URL for accessing the file
    """
    # Extract bucket and path from full storage path
    parts = storage_path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid storage path: {storage_path}")

    bucket = parts[0]
    path = parts[1]

    return get_signed_url(bucket, path, expires_in)


def upload_voice_sample(job_id: str, file_path: str, speaker_id: str = "default") -> str:
    """
    Upload a voice sample to Supabase Storage for Replicate API.

    The voice sample is uploaded to a temporary bucket that Replicate can
    access via signed URL for voice cloning.

    Args:
        job_id: UUID of the job (preview or translation)
        file_path: Local path to the WAV file
        speaker_id: Speaker identifier (for multi-speaker support)

    Returns:
        Storage path of the uploaded file

    Raises:
        Exception: If upload fails
    """
    return upload_to_storage(
        bucket=VOICE_SAMPLES_BUCKET,
        job_id=job_id,
        file_path=file_path,
        filename=f"{speaker_id}.wav",
        content_type="audio/wav"
    )


def get_voice_sample_signed_url(storage_path: str, expires_in: int = 3600) -> str:
    """
    Get a signed URL for a voice sample file.

    Used to provide Replicate API with access to the voice sample.

    Args:
        storage_path: Full storage path (e.g., "voice-samples/{job_id}/{speaker_id}.wav")
        expires_in: URL expiration time in seconds (default: 1 hour)

    Returns:
        Signed URL for accessing the file
    """
    # Extract bucket and path from full storage path
    parts = storage_path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid storage path: {storage_path}")

    bucket = parts[0]
    path = parts[1]

    return get_signed_url(bucket, path, expires_in)


def upload_temp_file(file_path: str, job_id: str = None, expires_in: int = 3600) -> str:
    """
    Upload a temporary file to Supabase Storage and return a signed URL.

    Used for passing large audio/video files to RunPod endpoints via URL
    instead of base64 encoding (which has size limits).

    Args:
        file_path: Local path to the file to upload
        job_id: Optional job ID for organizing files. If not provided,
                uses a timestamp-based ID.
        expires_in: URL expiration time in seconds (default: 1 hour)

    Returns:
        Signed URL for accessing the uploaded file

    Raises:
        Exception: If upload fails
    """
    import uuid
    from pathlib import Path

    # Generate job_id if not provided
    if job_id is None:
        job_id = str(uuid.uuid4())

    # Determine content type from extension
    file_ext = Path(file_path).suffix.lower()
    content_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    content_type = content_types.get(file_ext, "application/octet-stream")

    # Get filename from path
    filename = Path(file_path).name

    # Upload to temp bucket
    storage_path = upload_to_storage(
        bucket=TEMP_BUCKET,
        job_id=job_id,
        file_path=file_path,
        filename=filename,
        content_type=content_type
    )

    # Get signed URL
    # storage_path is "temp-files/{job_id}/{filename}", we need just "{job_id}/{filename}"
    path_without_bucket = storage_path.split("/", 1)[1]
    return get_signed_url(TEMP_BUCKET, path_without_bucket, expires_in)


# --- Authentication Operations ---

def signup_user(email: str, password: str) -> dict:
    """
    Create a new user in Supabase Auth.

    Args:
        email: User's email address
        password: User's password

    Returns:
        Dict with user info and session tokens:
        {
            "user_id": str,
            "email": str,
            "access_token": str,
            "refresh_token": str
        }

    Raises:
        ValueError: If signup fails (email already exists, invalid email, weak password)
    """
    client = get_supabase_client()

    try:
        result = client.auth.sign_up({
            "email": email,
            "password": password
        })

        if result.user is None:
            raise ValueError("Signup failed: no user returned")

        if result.session is None:
            # User created but email confirmation required
            # Still return user info but without session tokens
            return {
                "user_id": result.user.id,
                "email": result.user.email or email,
                "access_token": "",
                "refresh_token": "",
                "email_confirmed": False
            }

        return {
            "user_id": result.user.id,
            "email": result.user.email or email,
            "access_token": result.session.access_token,
            "refresh_token": result.session.refresh_token,
            "email_confirmed": True
        }

    except Exception as e:
        error_msg = str(e).lower()
        if "already registered" in error_msg or "already exists" in error_msg:
            raise ValueError("Email already registered")
        elif "invalid" in error_msg and "email" in error_msg:
            raise ValueError("Invalid email format")
        elif "password" in error_msg:
            raise ValueError("Password does not meet requirements (minimum 6 characters)")
        else:
            raise ValueError(f"Signup failed: {str(e)}")


def login_user(email: str, password: str) -> dict:
    """
    Authenticate a user with email and password.

    Args:
        email: User's email address
        password: User's password

    Returns:
        Dict with user info and session tokens:
        {
            "user_id": str,
            "email": str,
            "access_token": str,
            "refresh_token": str
        }

    Raises:
        ValueError: If login fails (invalid credentials)
    """
    client = get_supabase_client()

    try:
        result = client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if result.user is None or result.session is None:
            raise ValueError("Invalid email or password")

        return {
            "user_id": result.user.id,
            "email": result.user.email or email,
            "access_token": result.session.access_token,
            "refresh_token": result.session.refresh_token
        }

    except Exception as e:
        error_msg = str(e).lower()
        if "invalid" in error_msg or "credentials" in error_msg or "password" in error_msg:
            raise ValueError("Invalid email or password")
        else:
            raise ValueError(f"Login failed: {str(e)}")


def get_google_oauth_url(redirect_url: str) -> str:
    """
    Get the Google OAuth URL for sign-in.

    Args:
        redirect_url: The callback URL to redirect to after OAuth completes

    Returns:
        The Google OAuth authorization URL

    Raises:
        ValueError: If OAuth URL generation fails
    """
    client = get_supabase_client()

    try:
        result = client.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": redirect_url
            }
        })

        if result.url is None:
            raise ValueError("Failed to get OAuth URL")

        return result.url

    except Exception as e:
        raise ValueError(f"Failed to get Google OAuth URL: {str(e)}")


def exchange_oauth_code(code: str) -> dict:
    """
    Exchange an OAuth code for session tokens.

    Args:
        code: The authorization code from the OAuth callback

    Returns:
        Dict with user info and session tokens:
        {
            "user_id": str,
            "email": str,
            "access_token": str,
            "refresh_token": str,
            "is_new_user": bool
        }

    Raises:
        ValueError: If code exchange fails
    """
    client = get_supabase_client()

    try:
        result = client.auth.exchange_code_for_session({"auth_code": code})

        if result.user is None or result.session is None:
            raise ValueError("Failed to exchange code for session")

        # Check if user is new by looking for their profile
        profile = get_profile(result.user.id)
        is_new_user = profile is None

        return {
            "user_id": result.user.id,
            "email": result.user.email or "",
            "access_token": result.session.access_token,
            "refresh_token": result.session.refresh_token,
            "is_new_user": is_new_user
        }

    except Exception as e:
        raise ValueError(f"Failed to exchange OAuth code: {str(e)}")


# --- Stripe Operations ---

def create_stripe_checkout_session(
    translation_job_id: str,
    price_cents: int,
    video_title: str,
    customer_email: Optional[str] = None
) -> dict:
    """
    Create a Stripe Checkout session for a translation job.

    Args:
        translation_job_id: UUID of the translation job (used for metadata)
        price_cents: Price in cents
        video_title: Video title for the line item description
        customer_email: Optional customer email for pre-filling checkout

    Returns:
        Dict with checkout session info:
        {
            "checkout_session_id": str,
            "checkout_url": str
        }

    Raises:
        ValueError: If Stripe is not configured or checkout creation fails
    """
    if not STRIPE_SECRET_KEY:
        raise ValueError("Stripe is not configured (STRIPE_SECRET_KEY not set)")

    try:
        # Create checkout session
        checkout_params: dict = {
            "payment_method_types": ["card"],
            "line_items": [{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": "Video Translation",
                        "description": f"Full translation of: {video_title[:100]}"
                    },
                    "unit_amount": price_cents,
                },
                "quantity": 1,
            }],
            "mode": "payment",
            "success_url": f"{STRIPE_SUCCESS_URL}?session_id={{CHECKOUT_SESSION_ID}}",
            "cancel_url": STRIPE_CANCEL_URL,
            "metadata": {
                "translation_job_id": translation_job_id
            }
        }

        if customer_email:
            checkout_params["customer_email"] = customer_email

        session = stripe.checkout.Session.create(**checkout_params)

        return {
            "checkout_session_id": session.id,
            "checkout_url": session.url or ""
        }

    except stripe.StripeError as e:
        raise ValueError(f"Stripe checkout creation failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to create checkout session: {str(e)}")


# --- Connection Test ---

def verify_jwt(access_token: str) -> dict:
    """
    Verify a Supabase JWT access token and extract user information.

    Uses Supabase's get_user() to validate the token server-side.
    This is the most reliable method as it validates against Supabase Auth.

    Args:
        access_token: JWT access token from client

    Returns:
        Dict with user info:
        {
            "user_id": str,
            "email": str
        }

    Raises:
        ValueError: If token is invalid or expired
    """
    client = get_supabase_client()

    try:
        # Use Supabase to verify the token and get user info
        result = client.auth.get_user(access_token)

        if result.user is None:
            raise ValueError("Invalid or expired token")

        return {
            "user_id": result.user.id,
            "email": result.user.email or ""
        }

    except Exception as e:
        error_msg = str(e).lower()
        if "invalid" in error_msg or "expired" in error_msg or "jwt" in error_msg:
            raise ValueError("Invalid or expired token")
        else:
            raise ValueError(f"Token verification failed: {str(e)}")


def verify_stripe_webhook_signature(payload: bytes, signature: str) -> dict:
    """
    Verify a Stripe webhook signature and parse the event.

    Args:
        payload: Raw request body bytes
        signature: Stripe-Signature header value

    Returns:
        Parsed Stripe event dict

    Raises:
        ValueError: If signature verification fails or webhook secret not configured
    """
    if not STRIPE_WEBHOOK_SECRET:
        raise ValueError("Stripe webhook secret not configured (STRIPE_WEBHOOK_SECRET not set)")

    try:
        event = stripe.Webhook.construct_event(
            payload, signature, STRIPE_WEBHOOK_SECRET
        )
        return event
    except stripe.SignatureVerificationError as e:
        raise ValueError(f"Invalid webhook signature: {str(e)}")
    except Exception as e:
        raise ValueError(f"Webhook verification failed: {str(e)}")


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
