"""
Unified Self-hosted GPU Handler for Dubbing Studio

FastAPI server for the AI pipeline with endpoints for:
- Health monitoring (/health)
- Future: Transcription, diarization, separation, translation, TTS, mixing

The server loads AI models on startup and reports their status.
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any

import psutil
from fastapi import FastAPI
from pydantic import BaseModel

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
# Global State
# =============================================================================


class AppState:
    """Application state for tracking loaded models."""

    def __init__(self) -> None:
        self.models_loaded: list[str] = []
        self.model_errors: dict[str, str] = {}
        self.shutdown_event: asyncio.Event | None = None


app_state = AppState()


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
    # Create model instance to verify loading works
    # In production, this would be stored in app_state for reuse
    _ = WhisperModel(model_size, device=device, compute_type=compute_type)


def load_demucs() -> None:
    """Load Demucs model for audio separation."""
    import torch

    logger.info("  → Loading Demucs htdemucs model")
    # Load from torch hub
    _ = torch.hub.load("facebookresearch/demucs", "htdemucs", source="github")


def load_pyannote() -> None:
    """Load pyannote.audio pipeline for speaker diarization."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable required for pyannote.audio")

    from pyannote.audio import Pipeline

    logger.info("  → Loading pyannote speaker-diarization-3.1 pipeline")
    _ = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )


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
