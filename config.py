"""
Configuration constants for the yt-translate application.

This module centralizes all magic numbers and hardcoded values,
making the codebase more maintainable and self-documenting.
"""

# ============================================================================
# Google Translate Constants
# ============================================================================
GOOGLE_TRANSLATE_CHUNK_SIZE = 4500  # Maximum characters per Google Translate request

# ============================================================================
# LLM Translation Constants (Replicate Llama)
# ============================================================================
LLM_MAX_SEGMENTS_PER_BATCH = 50  # Maximum transcript segments per LLM batch
LLM_BATCH_OVERLAP = 5  # Overlapping segments between batches for context continuity
LLM_MAX_TOKENS = 4096  # Maximum tokens for LLM response
LLM_TEMPERATURE = 0.3  # Lower temperature for more consistent translations

# ============================================================================
# Chatterbox TTS Constants
# ============================================================================
CHATTERBOX_SAMPLE_RATE = 24000  # Chatterbox requires 24kHz audio
CHATTERBOX_CFG_WEIGHT = 0.5  # CFG/Pace weight (0.0-1.0, higher = slower/more stable)
CHATTERBOX_EXAGGERATION = 0.5  # Voice exaggeration (0.0-1.0, higher = more expressive)

# ============================================================================
# Voice Sample Extraction Constants
# ============================================================================
VOICE_SAMPLE_DURATION_SECONDS = 10.0  # Duration of voice sample to extract for cloning
MIN_VOICE_SAMPLE_DURATION = 3.0  # Minimum segment duration required for extraction
MAX_VOICE_SAMPLE_DURATION = 12.0  # Maximum duration for each voice sample

# ============================================================================
# Signed URL Configuration
# ============================================================================
SIGNED_URL_EXPIRATION = 3600  # Default signed URL expiration (1 hour)
SIGNED_URL_EXPIRATION_LONG = 86400  # Long-lived signed URL expiration (24 hours)

# ============================================================================
# Audio Processing Constants
# ============================================================================
BACKGROUND_VOLUME_DEFAULT = 0.3  # Default volume level for background audio (30%)

# ============================================================================
# Replicate Model Names
# ============================================================================
REPLICATE_MODEL_LLAMA = "meta/meta-llama-3.1-405b-instruct"
REPLICATE_MODEL_CHATTERBOX = "resemble-ai/chatterbox-multilingual:9cfba4c265e685f840612be835424f8c33bdee685d7466ece7684b0d9d4c0b1c"
REPLICATE_MODEL_DEMUCS = "cjwbw/demucs:25a173108cff36ef9f80f854c162d01df9e6528be175794b81158fa03836d953"
REPLICATE_MODEL_DIARIZATION = "meronym/speaker-diarization:64b78c82f74d78164b49178443c819445f5dca2c51c8ec374783d49382342119"
