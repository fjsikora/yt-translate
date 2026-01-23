"""
Unified Self-hosted GPU Handler for Dubbing Studio

FastAPI server for the AI pipeline with endpoints for:
- Health monitoring (/health)
- Transcription (/transcribe)
- Future: Diarization, separation, translation, TTS, mixing

The server loads AI models on startup and reports their status.
"""

import asyncio
import logging
import os
import signal
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dubbing-studio")

# =============================================================================
# Response Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    models_loaded: list[str]
    gpu_memory_used: float | None
    gpu_memory_total: float | None
    system_memory_used: float
    system_memory_total: float


class ModelStatus(BaseModel):
    """Individual model status."""

    name: str
    loaded: bool
    error: str | None = None


# =============================================================================
# Transcription Models
# =============================================================================


class TranscribeRequest(BaseModel):
    """Request model for transcription endpoint."""

    audio_url: HttpUrl = Field(..., description="URL of audio file to transcribe")
    word_timestamps: bool = Field(
        default=False, description="Include word-level timestamps in response"
    )


class TranscriptionSegment(BaseModel):
    """A single transcription segment."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")


class TranscribeResponse(BaseModel):
    """Response model for transcription endpoint."""

    segments: list[TranscriptionSegment] = Field(
        default_factory=list, description="List of transcription segments"
    )
    language: str = Field(..., description="Detected language code")
    text: str = Field(default="", description="Full transcription text")
    duration: float = Field(default=0.0, description="Audio duration in seconds")


# =============================================================================
# Diarization Models
# =============================================================================


class DiarizeRequest(BaseModel):
    """Request model for speaker diarization endpoint."""

    audio_url: HttpUrl = Field(..., description="URL of audio file to diarize")


class DiarizationSegment(BaseModel):
    """A single speaker diarization segment."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    speaker: str = Field(..., description="Speaker label (e.g., SPEAKER_00)")


class DiarizeResponse(BaseModel):
    """Response model for speaker diarization endpoint."""

    segments: list[DiarizationSegment] = Field(
        default_factory=list, description="List of speaker segments"
    )


# =============================================================================
# Audio Separation Models
# =============================================================================


class SeparateRequest(BaseModel):
    """Request model for audio separation endpoint."""

    audio_url: HttpUrl = Field(..., description="URL of audio file to separate")


class SeparateResponse(BaseModel):
    """Response model for audio separation endpoint."""

    vocals_url: str = Field(..., description="Signed URL for vocals audio file")
    background_url: str = Field(..., description="Signed URL for background audio file")


# =============================================================================
# Global State
# =============================================================================


class AppState:
    """Application state for tracking loaded models."""

    def __init__(self) -> None:
        self.models_loaded: list[str] = []
        self.model_errors: dict[str, str] = {}
        self.shutdown_event: asyncio.Event | None = None
        # Model instances
        self.whisper_model: Any = None
        self.demucs_model: Any = None
        self.pyannote_pipeline: Any = None
        self.chatterbox_model: Any = None
        self.llama_model: Any = None


app_state = AppState()


# =============================================================================
# Audio Download Utility
# =============================================================================


async def download_audio(audio_url: str, timeout: float = 300.0) -> Path:
    """
    Download audio file from URL to a temporary location.

    Args:
        audio_url: URL of the audio file (e.g., Supabase signed URL)
        timeout: Download timeout in seconds

    Returns:
        Path to the downloaded temporary file

    Raises:
        HTTPException: If download fails
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(audio_url)
            response.raise_for_status()

            # Determine file extension from content-type or URL
            content_type = response.headers.get("content-type", "")
            if "wav" in content_type or audio_url.endswith(".wav"):
                suffix = ".wav"
            elif "mp3" in content_type or audio_url.endswith(".mp3"):
                suffix = ".mp3"
            elif "flac" in content_type or audio_url.endswith(".flac"):
                suffix = ".flac"
            elif "ogg" in content_type or audio_url.endswith(".ogg"):
                suffix = ".ogg"
            else:
                suffix = ".wav"  # Default to wav

            # Write to temp file
            temp_file = tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, prefix="audio_"
            )
            temp_file.write(response.content)
            temp_file.close()

            logger.info(
                f"Downloaded audio to {temp_file.name} ({len(response.content)} bytes)"
            )
            return Path(temp_file.name)

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading audio: {e.response.status_code}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download audio: HTTP {e.response.status_code}",
        )
    except httpx.RequestError as e:
        logger.error(f"Request error downloading audio: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to download audio: {e}")


# =============================================================================
# GPU Memory Utilities
# =============================================================================


def get_gpu_memory() -> tuple[float | None, float | None]:
    """
    Get GPU memory usage in GB.
    Returns (used_gb, total_gb) or (None, None) if GPU not available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            # Get memory stats for the current device
            device = torch.cuda.current_device()
            used = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GB
            total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            return round(used, 2), round(total, 2)
    except ImportError:
        logger.warning("PyTorch not available, cannot get GPU memory stats")
    except Exception as e:
        logger.warning(f"Error getting GPU memory: {e}")
    return None, None


def get_system_memory() -> tuple[float, float]:
    """Get system memory usage in GB. Returns (used_gb, total_gb)."""
    mem = psutil.virtual_memory()
    used = mem.used / (1024**3)
    total = mem.total / (1024**3)
    return round(used, 2), round(total, 2)


# =============================================================================
# Model Loading
# =============================================================================


async def load_models() -> None:
    """
    Load all AI models on startup.
    This is called during the lifespan startup phase.
    """
    models_to_load = [
        ("whisper", load_whisper),
        ("demucs", load_demucs),
        ("pyannote", load_pyannote),
        ("chatterbox", load_chatterbox),
        ("llama", load_llama),
    ]

    for model_name, loader in models_to_load:
        logger.info(f"Loading model: {model_name}...")
        try:
            # Run model loading in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(None, loader)
            app_state.models_loaded.append(model_name)
            logger.info(f"✓ Model loaded: {model_name}")
        except Exception as e:
            error_msg = str(e)
            app_state.model_errors[model_name] = error_msg
            logger.error(f"✗ Failed to load {model_name}: {error_msg}")

    # Log summary
    loaded_count = len(app_state.models_loaded)
    total_count = len(models_to_load)
    logger.info(f"Model loading complete: {loaded_count}/{total_count} models loaded")

    if app_state.model_errors:
        logger.warning(f"Models with errors: {list(app_state.model_errors.keys())}")


def load_whisper() -> None:
    """Load Whisper model for transcription."""
    # Only import when loading to avoid startup delays if model load fails
    from faster_whisper import WhisperModel

    model_size = os.getenv("WHISPER_MODEL", "large-v3")
    device = "cuda" if is_cuda_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info(f"  → Whisper model: {model_size}, device: {device}")
    # Create model instance and store in app_state for reuse
    app_state.whisper_model = WhisperModel(
        model_size, device=device, compute_type=compute_type
    )


def load_demucs() -> None:
    """Load Demucs model for audio separation."""
    from demucs.pretrained import get_model

    logger.info("  → Loading Demucs htdemucs model")
    model = get_model("htdemucs")
    model.eval()

    # Move to GPU if available
    if is_cuda_available():
        import torch

        model.to(torch.device("cuda"))
        logger.info("  → Demucs model moved to GPU")

    app_state.demucs_model = model


def load_pyannote() -> None:
    """Load pyannote.audio pipeline for speaker diarization."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable required for pyannote.audio")

    from pyannote.audio import Pipeline

    logger.info("  → Loading pyannote speaker-diarization-3.1 pipeline")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )

    # Move to GPU if available
    if is_cuda_available():
        import torch

        pipeline.to(torch.device("cuda"))
        logger.info("  → Pyannote pipeline moved to GPU")

    app_state.pyannote_pipeline = pipeline


def load_chatterbox() -> None:
    """Load Chatterbox TTS model for voice cloning."""
    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if is_cuda_available() else "cpu"
    logger.info(f"  → Loading Chatterbox TTS, device: {device}")
    _ = ChatterboxTTS.from_pretrained(device=device)


def load_llama() -> None:
    """Load Qwen3 model via llama.cpp for translation."""
    from llama_cpp import Llama

    model_path = os.getenv("MODEL_PATH", "/models/qwen3-8b-q4_k_m.gguf")
    if not os.path.exists(model_path):
        raise RuntimeError(f"GGUF model not found at {model_path}")

    n_gpu_layers = -1 if is_cuda_available() else 0  # -1 = all layers on GPU
    logger.info(f"  → Loading Qwen3-8B from {model_path}, GPU layers: {n_gpu_layers}")

    _ = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=8192,
        verbose=False,
    )


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# =============================================================================
# Graceful Shutdown
# =============================================================================


def setup_signal_handlers(shutdown_event: asyncio.Event) -> None:
    """Set up signal handlers for graceful shutdown."""

    def handle_signal(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        shutdown_event.set()

    # Register handlers for SIGTERM and SIGINT
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)


# =============================================================================
# Application Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """
    Application lifespan manager.
    Handles startup (model loading) and shutdown (cleanup).
    """
    logger.info("=" * 60)
    logger.info("Starting Dubbing Studio API Server")
    logger.info("=" * 60)

    # Set up shutdown event
    app_state.shutdown_event = asyncio.Event()
    setup_signal_handlers(app_state.shutdown_event)

    # Load models on startup
    await load_models()

    # Log GPU status
    gpu_used, gpu_total = get_gpu_memory()
    if gpu_total:
        logger.info(f"GPU Memory: {gpu_used:.2f}GB / {gpu_total:.2f}GB")
    else:
        logger.info("No GPU detected - running in CPU mode")

    logger.info("Server ready to accept requests")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down Dubbing Studio API Server")
    logger.info("=" * 60)

    # Cleanup resources (future: release model memory)
    logger.info("Cleanup complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Dubbing Studio API",
    description="Unified AI pipeline for video dubbing",
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse with status, loaded models, and memory usage.
    """
    gpu_used, gpu_total = get_gpu_memory()
    sys_used, sys_total = get_system_memory()

    # Determine overall status
    # If any critical models failed, report degraded
    critical_models = {"whisper", "demucs", "pyannote", "chatterbox", "llama"}
    loaded_set = set(app_state.models_loaded)
    missing_critical = critical_models - loaded_set

    if not missing_critical:
        status = "healthy"
    elif len(app_state.models_loaded) > 0:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        models_loaded=app_state.models_loaded,
        gpu_memory_used=gpu_used,
        gpu_memory_total=gpu_total,
        system_memory_used=sys_used,
        system_memory_total=sys_total,
    )


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "Dubbing Studio API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest) -> TranscribeResponse:
    """
    Transcribe audio file using Whisper.

    Accepts an audio URL (e.g., from Supabase storage) and returns
    timestamped transcription segments.

    Args:
        request: TranscribeRequest with audio_url and optional word_timestamps

    Returns:
        TranscribeResponse with segments, detected language, and full text

    Raises:
        HTTPException: If transcription fails or Whisper model not loaded
    """
    # Check if Whisper model is loaded
    if app_state.whisper_model is None:
        raise HTTPException(
            status_code=503,
            detail="Whisper model not loaded. Check /health for model status.",
        )

    temp_file: Path | None = None

    try:
        # Download audio from URL
        logger.info(f"Transcribing audio from: {request.audio_url}")
        temp_file = await download_audio(str(request.audio_url))

        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        segments_data, info = await loop.run_in_executor(
            None,
            _run_whisper_transcription,
            str(temp_file),
            request.word_timestamps,
        )

        # Build response
        segments = [
            TranscriptionSegment(start=seg["start"], end=seg["end"], text=seg["text"])
            for seg in segments_data
        ]
        full_text = " ".join(seg["text"] for seg in segments_data)

        logger.info(
            f"Transcription complete: {len(segments)} segments, "
            f"language={info['language']}, duration={info['duration']:.1f}s"
        )

        return TranscribeResponse(
            segments=segments,
            language=info["language"],
            text=full_text,
            duration=info["duration"],
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        # Clean up temp file
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")


def _run_whisper_transcription(
    audio_path: str, word_timestamps: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run Whisper transcription synchronously.

    This function is meant to be run in a thread pool executor.

    Args:
        audio_path: Path to the audio file
        word_timestamps: Whether to include word-level timestamps

    Returns:
        Tuple of (segments list, info dict with language and duration)
    """
    segments_list: list[dict[str, Any]] = []

    # Run transcription
    segments_generator, info = app_state.whisper_model.transcribe(
        audio_path,
        word_timestamps=word_timestamps,
        vad_filter=True,  # Voice activity detection for better quality
    )

    # Consume the generator and build segments list
    for segment in segments_generator:
        segment_data: dict[str, Any] = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        }

        # Include word timestamps if requested
        if word_timestamps and segment.words:
            segment_data["words"] = [
                {"start": word.start, "end": word.end, "word": word.word}
                for word in segment.words
            ]

        segments_list.append(segment_data)

    return segments_list, {"language": info.language, "duration": info.duration}


@app.post("/diarize", response_model=DiarizeResponse)
async def diarize(request: DiarizeRequest) -> DiarizeResponse:
    """
    Perform speaker diarization on audio file using pyannote.audio.

    Identifies different speakers in the audio and returns timestamped segments
    with speaker labels. Handles overlapping speech by assigning the most
    prominent speaker to each time window.

    Args:
        request: DiarizeRequest with audio_url

    Returns:
        DiarizeResponse with segments containing start, end, and speaker label

    Raises:
        HTTPException: If diarization fails or pyannote pipeline not loaded
    """
    # Check if pyannote pipeline is loaded
    if app_state.pyannote_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pyannote pipeline not loaded. Check /health for model status.",
        )

    temp_file: Path | None = None

    try:
        # Download audio from URL
        logger.info(f"Diarizing audio from: {request.audio_url}")
        temp_file = await download_audio(str(request.audio_url))

        # Run diarization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        segments_data = await loop.run_in_executor(
            None,
            _run_pyannote_diarization,
            str(temp_file),
        )

        # Build response
        segments = [
            DiarizationSegment(
                start=seg["start"], end=seg["end"], speaker=seg["speaker"]
            )
            for seg in segments_data
        ]

        # Count unique speakers
        unique_speakers = set(seg["speaker"] for seg in segments_data)
        logger.info(
            f"Diarization complete: {len(segments)} segments, "
            f"{len(unique_speakers)} speakers detected"
        )

        return DiarizeResponse(segments=segments)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Diarization failed: {e}")
    finally:
        # Clean up temp file
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")


def _run_pyannote_diarization(audio_path: str) -> list[dict[str, Any]]:
    """
    Run pyannote diarization synchronously.

    This function is meant to be run in a thread pool executor.

    Args:
        audio_path: Path to the audio file

    Returns:
        List of segment dicts with 'start', 'end', 'speaker' keys
    """
    # Run diarization (pyannote 3.x returns DiarizeOutput dataclass)
    output = app_state.pyannote_pipeline(audio_path)

    # Handle different pyannote output formats
    # pyannote 3.x returns DiarizeOutput with speaker_diarization attribute
    # pyannote 2.x returns Annotation directly
    if hasattr(output, "speaker_diarization"):
        diarization = output.speaker_diarization
    else:
        diarization = output

    # Convert to list of segments
    segments_list: list[dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments_list.append(
            {
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            }
        )

    # Sort by start time to ensure chronological order
    segments_list.sort(key=lambda x: x["start"])

    return segments_list


@app.post("/separate", response_model=SeparateResponse)
async def separate(request: SeparateRequest) -> SeparateResponse:
    """
    Separate audio into vocals and background using Demucs.

    Accepts an audio URL (e.g., from Supabase storage) and returns
    signed URLs for the separated vocals and background audio files.

    Args:
        request: SeparateRequest with audio_url

    Returns:
        SeparateResponse with vocals_url and background_url (Supabase signed URLs)

    Raises:
        HTTPException: If separation fails or Demucs model not loaded
    """
    # Check if Demucs model is loaded
    if app_state.demucs_model is None:
        raise HTTPException(
            status_code=503,
            detail="Demucs model not loaded. Check /health for model status.",
        )

    temp_file: Path | None = None
    vocals_path: Path | None = None
    background_path: Path | None = None

    try:
        # Download audio from URL
        logger.info(f"Separating audio from: {request.audio_url}")
        temp_file = await download_audio(str(request.audio_url))

        # Run separation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        vocals_array, background_array, sample_rate = await loop.run_in_executor(
            None,
            _run_demucs_separation,
            str(temp_file),
        )

        # Save separated audio to temp files
        vocals_path = Path(tempfile.mktemp(suffix=".wav", prefix="vocals_"))
        background_path = Path(tempfile.mktemp(suffix=".wav", prefix="background_"))

        import soundfile as sf

        sf.write(str(vocals_path), vocals_array, sample_rate)
        sf.write(str(background_path), background_array, sample_rate)

        logger.info(
            f"Separation complete: vocals={vocals_path.stat().st_size / 1e6:.1f}MB, "
            f"background={background_path.stat().st_size / 1e6:.1f}MB"
        )

        # Upload to Supabase storage
        vocals_url, background_url = await _upload_separated_audio(
            vocals_path, background_path
        )

        return SeparateResponse(
            vocals_url=vocals_url,
            background_url=background_url,
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Audio separation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio separation failed: {e}")
    finally:
        # Clean up temp files
        for path in [temp_file, vocals_path, background_path]:
            if path and path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")


def _run_demucs_separation(audio_path: str) -> tuple[Any, Any, int]:
    """
    Run Demucs audio separation synchronously.

    This function is meant to be run in a thread pool executor.

    Args:
        audio_path: Path to the audio file

    Returns:
        Tuple of (vocals_array, background_array, sample_rate)
    """
    import soundfile as sf
    import torch
    import torchaudio
    from demucs.apply import apply_model

    model = app_state.demucs_model

    # Load audio with soundfile
    audio_data, input_sr = sf.read(audio_path)
    wav = torch.from_numpy(audio_data).float()

    # Handle shape: soundfile returns (samples,) or (samples, channels)
    # torch expects (channels, samples)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # (samples,) -> (1, samples)
    elif wav.dim() == 2:
        wav = wav.T  # (samples, channels) -> (channels, samples)

    # Get model's expected sample rate (htdemucs uses 44100)
    model_sr = model.samplerate

    # Resample to model's sample rate if needed
    if input_sr != model_sr:
        logger.info(f"Resampling from {input_sr} to {model_sr} Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=input_sr, new_freq=model_sr
        )
        wav = resampler(wav)

    # Ensure stereo (Demucs expects stereo input)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    # Add batch dimension and move to device
    device = next(model.parameters()).device
    wav = wav.unsqueeze(0).to(device)

    # Apply model
    logger.info("Running Demucs source separation...")
    with torch.no_grad():
        sources = apply_model(model, wav, device=device, progress=False)

    # sources shape: (batch, sources, channels, samples)
    # htdemucs sources: drums, bass, other, vocals
    sources = sources.squeeze(0).cpu()

    # Get source names
    source_names = model.sources
    vocals_idx = source_names.index("vocals") if "vocals" in source_names else -1

    if vocals_idx == -1:
        raise RuntimeError("Model doesn't have 'vocals' source")

    # Extract vocals and combine others for background
    vocals = sources[vocals_idx]
    background_sources = [
        sources[i] for i in range(len(source_names)) if i != vocals_idx
    ]
    background = sum(background_sources)

    # Convert to mono and numpy
    vocals_array = vocals.mean(dim=0).numpy()
    background_array = background.mean(dim=0).numpy()

    # Clean up GPU memory
    del wav, sources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vocals_array, background_array, model_sr


async def _upload_separated_audio(
    vocals_path: Path, background_path: Path
) -> tuple[str, str]:
    """
    Upload separated audio files to Supabase storage.

    Args:
        vocals_path: Path to vocals audio file
        background_path: Path to background audio file

    Returns:
        Tuple of (vocals_signed_url, background_signed_url)
    """
    import uuid

    from supabase import create_client

    # Get Supabase credentials from environment
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables required"
        )

    client = create_client(supabase_url, supabase_key)

    # Generate unique job ID for storage path
    job_id = str(uuid.uuid4())
    bucket = "dub-audio"  # Bucket for dubbing audio files

    # Upload vocals
    vocals_storage_path = f"{job_id}/vocals.wav"
    with open(vocals_path, "rb") as f:
        vocals_content = f.read()
    client.storage.from_(bucket).upload(
        path=vocals_storage_path,
        file=vocals_content,
        file_options={"content-type": "audio/wav", "upsert": "true"},
    )

    # Upload background
    background_storage_path = f"{job_id}/background.wav"
    with open(background_path, "rb") as f:
        background_content = f.read()
    client.storage.from_(bucket).upload(
        path=background_storage_path,
        file=background_content,
        file_options={"content-type": "audio/wav", "upsert": "true"},
    )

    # Get signed URLs (1 hour expiry)
    vocals_signed = client.storage.from_(bucket).create_signed_url(
        vocals_storage_path, 3600
    )
    background_signed = client.storage.from_(bucket).create_signed_url(
        background_storage_path, 3600
    )

    logger.info(f"Uploaded separated audio to Supabase: {job_id}")

    return vocals_signed.get("signedURL", ""), background_signed.get("signedURL", "")
