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
# Audio Mixing Models
# =============================================================================


class MixRequest(BaseModel):
    """Request model for audio mixing endpoint."""

    speech_url: HttpUrl = Field(..., description="URL of speech audio file")
    background_url: HttpUrl = Field(..., description="URL of background audio file")
    background_volume: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Background volume level (0.0 = silent, 1.0 = full volume)",
    )


class MixResponse(BaseModel):
    """Response model for audio mixing endpoint."""

    mixed_url: str = Field(..., description="Signed URL for mixed audio file")


# =============================================================================
# TTS Synthesis Models
# =============================================================================


class SynthesizeRequest(BaseModel):
    """Request model for TTS synthesis endpoint."""

    text: str = Field(..., description="Text to synthesize into speech")
    voice_sample_url: HttpUrl = Field(
        ..., description="URL of voice sample for cloning"
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speed factor (0.5 = half speed, 2.0 = double speed)",
    )


class SynthesizeResponse(BaseModel):
    """Response model for TTS synthesis endpoint."""

    audio_url: str = Field(..., description="Signed URL for synthesized audio file")


# =============================================================================
# Translation Models
# =============================================================================


class TranslationSegment(BaseModel):
    """Input segment for translation."""

    index: int = Field(..., description="Segment index")
    text: str = Field(..., description="Text to translate")


class TranslateRequest(BaseModel):
    """Request model for translation endpoint."""

    segments: list[TranslationSegment] = Field(
        ..., description="List of segments to translate"
    )
    target_language: str = Field(
        ..., description="Target language name (e.g., 'Spanish')"
    )


class TranslationResult(BaseModel):
    """Single translation result."""

    index: int = Field(..., description="Original segment index")
    text: str = Field(..., description="Translated text")


class TranslateResponse(BaseModel):
    """Response model for translation endpoint."""

    translations: list[TranslationResult] = Field(
        default_factory=list, description="List of translations with indices"
    )


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
    app_state.chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
    logger.info(f"  → Chatterbox sample rate: {app_state.chatterbox_model.sr} Hz")


def load_llama() -> None:
    """Load Qwen3 model via llama.cpp for translation."""
    from llama_cpp import Llama

    model_path = os.getenv("MODEL_PATH", "/models/qwen3-8b-q4_k_m.gguf")
    if not os.path.exists(model_path):
        raise RuntimeError(f"GGUF model not found at {model_path}")

    n_gpu_layers = -1 if is_cuda_available() else 0  # -1 = all layers on GPU
    logger.info(f"  → Loading Qwen3-8B from {model_path}, GPU layers: {n_gpu_layers}")

    app_state.llama_model = Llama(
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


# =============================================================================
# Translation Endpoint
# =============================================================================

# Translation constants
TRANSLATION_MAX_SEGMENTS_PER_BATCH = 20
TRANSLATION_TEMPERATURE = 0.3
TRANSLATION_MAX_TOKENS = 4096


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest) -> TranslateResponse:
    """
    Translate text segments using Qwen3-8B via llama.cpp.

    Accepts a list of text segments and returns translations in the target language.
    Segments are batched (max 20 per batch) for optimal context handling.

    Args:
        request: TranslateRequest with segments list and target_language

    Returns:
        TranslateResponse with translations list containing index and translated text

    Raises:
        HTTPException: If translation fails or LLM model not loaded
    """
    # Check if LLM model is loaded
    if app_state.llama_model is None:
        raise HTTPException(
            status_code=503,
            detail="LLM model not loaded. Check /health for model status.",
        )

    if not request.segments:
        return TranslateResponse(translations=[])

    try:
        logger.info(
            f"Translating {len(request.segments)} segments to {request.target_language}"
        )

        # Process in batches
        all_translations: dict[int, str] = {}

        if len(request.segments) <= TRANSLATION_MAX_SEGMENTS_PER_BATCH:
            # Single batch - translate all at once
            transcript_lines = [f"{seg.index}: {seg.text}" for seg in request.segments]
            loop = asyncio.get_event_loop()
            batch_translations = await loop.run_in_executor(
                None,
                _run_llama_translation,
                transcript_lines,
                request.target_language,
            )
            all_translations = batch_translations
            logger.info(
                f"Translated {len(batch_translations)} segments in single batch"
            )
        else:
            # Multiple batches with context overlap
            total_batches = (
                len(request.segments) + TRANSLATION_MAX_SEGMENTS_PER_BATCH - 1
            ) // TRANSLATION_MAX_SEGMENTS_PER_BATCH
            logger.info(f"Processing {total_batches} batches")

            for batch_num in range(total_batches):
                start_idx = batch_num * TRANSLATION_MAX_SEGMENTS_PER_BATCH
                end_idx = min(
                    start_idx + TRANSLATION_MAX_SEGMENTS_PER_BATCH,
                    len(request.segments),
                )
                batch_segments = request.segments[start_idx:end_idx]

                transcript_lines = [
                    f"{seg.index}: {seg.text}" for seg in batch_segments
                ]

                loop = asyncio.get_event_loop()
                batch_translations = await loop.run_in_executor(
                    None,
                    _run_llama_translation,
                    transcript_lines,
                    request.target_language,
                )
                all_translations.update(batch_translations)
                logger.info(
                    f"  Batch {batch_num + 1}/{total_batches}: "
                    f"translated {len(batch_translations)} segments"
                )

        # Build response - maintain original segment order
        translations = [
            TranslationResult(index=seg.index, text=all_translations.get(seg.index, ""))
            for seg in request.segments
        ]

        # Log any missing translations
        missing = [
            seg.index for seg in request.segments if seg.index not in all_translations
        ]
        if missing:
            logger.warning(f"Missing translations for indices: {missing}")

        logger.info(f"Translation complete: {len(translations)} segments")
        return TranslateResponse(translations=translations)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")


def _run_llama_translation(
    transcript_lines: list[str], target_language: str
) -> dict[int, str]:
    """
    Run LLM translation synchronously.

    This function is meant to be run in a thread pool executor.

    Args:
        transcript_lines: List of "INDEX: text" formatted lines
        target_language: Human-readable language name (e.g., "Spanish")

    Returns:
        Dict mapping segment index to translated text
    """
    transcript = "\n".join(transcript_lines)

    prompt = f"""Translate the following numbered transcript lines to {target_language}.

IMPORTANT RULES:
1. Maintain the exact same line numbers in your output
2. Only output the translated text, preserving the "NUMBER: text" format
3. Keep translations natural and conversational
4. Maintain consistent terminology throughout
5. Preserve the speaker's tone and style
6. Do not add any explanations or notes

Transcript:
{transcript}

Translate each line to {target_language}, keeping the same numbered format:"""

    # Run inference
    output = app_state.llama_model(
        prompt,
        max_tokens=TRANSLATION_MAX_TOKENS,
        temperature=TRANSLATION_TEMPERATURE,
        stop=["</s>", "\n\n\n"],  # Stop tokens for Qwen
    )

    # Extract response text
    response_text = output.get("choices", [{}])[0].get("text", "")

    # Parse the numbered response back into translations
    translations: dict[int, str] = {}
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue

        parts = line.split(":", 1)
        try:
            idx_str = parts[0].strip()
            idx = int(idx_str)
            translations[idx] = parts[1].strip()
        except ValueError:
            continue

    return translations


# =============================================================================
# TTS Synthesis Endpoint
# =============================================================================

# TTS constants
TTS_SAMPLE_RATE = 24000  # Chatterbox native sample rate


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    """
    Synthesize speech from text using Chatterbox TTS with voice cloning.

    Accepts text and a voice sample URL for cloning, returns a signed URL
    for the synthesized audio file. Supports speed adjustment (0.5-2.0).

    Args:
        request: SynthesizeRequest with text, voice_sample_url, and optional speed

    Returns:
        SynthesizeResponse with audio_url (Supabase signed URL)

    Raises:
        HTTPException: If synthesis fails or Chatterbox model not loaded
    """
    # Check if Chatterbox model is loaded
    if app_state.chatterbox_model is None:
        raise HTTPException(
            status_code=503,
            detail="Chatterbox model not loaded. Check /health for model status.",
        )

    voice_sample_path: Path | None = None
    output_path: Path | None = None
    speed_adjusted_path: Path | None = None

    try:
        # Download voice sample
        logger.info(f"Synthesizing speech: '{request.text[:50]}...'")
        voice_sample_path = await download_audio(str(request.voice_sample_url))

        # Run TTS in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        audio_array, sample_rate = await loop.run_in_executor(
            None,
            _run_chatterbox_synthesis,
            request.text,
            str(voice_sample_path),
        )

        # Save synthesized audio to temp file
        output_path = Path(tempfile.mktemp(suffix=".wav", prefix="tts_"))

        import soundfile as sf

        sf.write(str(output_path), audio_array, sample_rate)

        logger.info(f"Synthesis complete: {len(audio_array) / sample_rate:.1f}s audio")

        # Apply speed adjustment if needed
        final_path = output_path
        if request.speed != 1.0:
            speed_adjusted_path = Path(
                tempfile.mktemp(suffix=".wav", prefix="tts_speed_")
            )
            await _apply_speed_adjustment(
                output_path, speed_adjusted_path, request.speed
            )
            final_path = speed_adjusted_path
            logger.info(f"Applied speed factor: {request.speed}x")

        # Upload to Supabase storage
        audio_url = await _upload_synthesized_audio(final_path)

        return SynthesizeResponse(audio_url=audio_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e}")
    finally:
        # Clean up temp files
        for path in [voice_sample_path, output_path, speed_adjusted_path]:
            if path and path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")


def _run_chatterbox_synthesis(text: str, voice_sample_path: str) -> tuple[Any, int]:
    """
    Run Chatterbox TTS synthesis synchronously.

    This function is meant to be run in a thread pool executor.

    Args:
        text: Text to synthesize
        voice_sample_path: Path to voice sample for cloning

    Returns:
        Tuple of (audio_array as numpy, sample_rate)
    """
    import numpy as np
    import torch

    model = app_state.chatterbox_model

    # Generate audio with voice cloning
    wav = model.generate(
        text,
        audio_prompt_path=voice_sample_path,
    )

    # Convert tensor to numpy array
    if torch.is_tensor(wav):
        audio_np = wav.squeeze().cpu().numpy()
    else:
        audio_np = np.array(wav).squeeze()

    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio_np))
    if max_val > 0.99:
        audio_np = audio_np * 0.99 / max_val

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return audio_np, model.sr


async def _apply_speed_adjustment(
    input_path: Path, output_path: Path, speed: float
) -> None:
    """
    Apply speed adjustment to audio using FFmpeg.

    Uses the atempo filter which preserves pitch while changing speed.
    FFmpeg's atempo only supports 0.5-2.0 range per filter instance.

    Args:
        input_path: Path to input audio file
        output_path: Path for output audio file
        speed: Speed factor (0.5 = half speed, 2.0 = double speed)
    """
    # FFmpeg atempo filter accepts values between 0.5 and 2.0
    # For our use case, this range is sufficient
    atempo_value = max(0.5, min(2.0, speed))

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i",
        str(input_path),
        "-filter:a",
        f"atempo={atempo_value}",
        "-acodec",
        "pcm_s16le",  # WAV format
        str(output_path),
    ]

    # Run FFmpeg
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg speed adjustment failed: {stderr.decode()}")


async def _upload_synthesized_audio(audio_path: Path) -> str:
    """
    Upload synthesized audio to Supabase storage.

    Args:
        audio_path: Path to audio file to upload

    Returns:
        Signed URL for the uploaded audio file
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

    # Generate unique path for storage
    job_id = str(uuid.uuid4())
    bucket = "dub-audio"  # Bucket for dubbing audio files
    storage_path = f"{job_id}/synthesized.wav"

    # Upload audio
    with open(audio_path, "rb") as f:
        audio_content = f.read()
    client.storage.from_(bucket).upload(
        path=storage_path,
        file=audio_content,
        file_options={"content-type": "audio/wav", "upsert": "true"},
    )

    # Get signed URL (1 hour expiry)
    signed_result = client.storage.from_(bucket).create_signed_url(storage_path, 3600)

    logger.info(f"Uploaded synthesized audio to Supabase: {job_id}")

    return signed_result.get("signedURL", "")


# =============================================================================
# Audio Mixing Endpoint
# =============================================================================


@app.post("/mix", response_model=MixResponse)
async def mix(request: MixRequest) -> MixResponse:
    """
    Mix speech audio with background audio using FFmpeg.

    Accepts speech and background audio URLs and returns a mixed audio file
    where the background is adjusted to the specified volume level.
    If audio tracks have different lengths, the shorter one is padded with silence.

    Args:
        request: MixRequest with speech_url, background_url, and background_volume

    Returns:
        MixResponse with mixed_url (Supabase signed URL)

    Raises:
        HTTPException: If mixing fails
    """
    speech_path: Path | None = None
    background_path: Path | None = None
    mixed_path: Path | None = None

    try:
        # Download both audio files
        logger.info(
            f"Mixing audio: speech + background (volume: {request.background_volume})"
        )

        speech_path = await download_audio(str(request.speech_url))
        background_path = await download_audio(str(request.background_url))

        # Create output path
        mixed_path = Path(tempfile.mktemp(suffix=".wav", prefix="mixed_"))

        # Run FFmpeg mixing
        await _run_ffmpeg_mix(
            speech_path=speech_path,
            background_path=background_path,
            output_path=mixed_path,
            background_volume=request.background_volume,
        )

        # Verify output
        if not mixed_path.exists():
            raise RuntimeError("FFmpeg mixing produced no output file")

        logger.info(f"Mixing complete: {mixed_path.stat().st_size / 1e6:.1f}MB")

        # Upload to Supabase storage
        mixed_url = await _upload_mixed_audio(mixed_path)

        return MixResponse(mixed_url=mixed_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio mixing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio mixing failed: {e}")
    finally:
        # Clean up temp files
        for path in [speech_path, background_path, mixed_path]:
            if path and path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")


async def _run_ffmpeg_mix(
    speech_path: Path,
    background_path: Path,
    output_path: Path,
    background_volume: float,
) -> None:
    """
    Mix speech and background audio using FFmpeg.

    Uses the amix filter to combine tracks. The background is padded with silence
    if shorter than speech, and the output duration matches the longer input.

    Filter breakdown:
    - [1:a]apad: Pad background with silence if shorter than speech
    - volume={vol}: Apply volume adjustment to background
    - amix: Mix the two streams together
    - duration=longest: Output matches the longest input

    Args:
        speech_path: Path to speech audio file
        background_path: Path to background audio file
        output_path: Path for output mixed audio file
        background_volume: Volume level for background (0.0 to 1.0)
    """
    # Build the FFmpeg filter for mixing
    # [1:a]apad: Pad background with silence if shorter than speech
    # volume={vol}: Apply volume to background
    # amix: Mix the two streams
    # duration=longest: Output duration matches the longest input
    filter_complex = (
        f"[1:a]apad,volume={background_volume}[bg];"
        f"[0:a][bg]amix=inputs=2:duration=longest:dropout_transition=0"
    )

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i",
        str(speech_path),
        "-i",
        str(background_path),
        "-filter_complex",
        filter_complex,
        "-ac",
        "2",  # Stereo output
        "-acodec",
        "pcm_s16le",  # WAV format
        str(output_path),
    ]

    logger.info(f"Running FFmpeg mix: volume={background_volume}")

    # Run FFmpeg
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode()[:500]
        raise RuntimeError(f"FFmpeg mixing failed: {error_msg}")


async def _upload_mixed_audio(audio_path: Path) -> str:
    """
    Upload mixed audio to Supabase storage.

    Args:
        audio_path: Path to audio file to upload

    Returns:
        Signed URL for the uploaded audio file
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

    # Generate unique path for storage
    job_id = str(uuid.uuid4())
    bucket = "dub-audio"  # Bucket for dubbing audio files
    storage_path = f"{job_id}/mixed.wav"

    # Upload audio
    with open(audio_path, "rb") as f:
        audio_content = f.read()
    client.storage.from_(bucket).upload(
        path=storage_path,
        file=audio_content,
        file_options={"content-type": "audio/wav", "upsert": "true"},
    )

    # Get signed URL (1 hour expiry)
    signed_result = client.storage.from_(bucket).create_signed_url(storage_path, 3600)

    logger.info(f"Uploaded mixed audio to Supabase: {job_id}")

    return signed_result.get("signedURL", "")
