# Archived Code

This directory contains archived code from previous implementations that have been superseded by newer approaches.

## Contents

### cloud_translate.py
The original cloud-based translation implementation using:
- **Replicate API** - For running Whisper transcription in the cloud
- **OpenAI API** - For GPT-based translation assistance

This was archived when the project migrated to **self-hosted RunPod Serverless** with local model inference.

### requirements-cloud.txt
The dependencies file for the cloud-based Railway API gateway, including:
- FastAPI web framework
- Replicate and OpenAI API clients
- Stripe payment integration
- Supabase database client

### Dated Subdirectories (YYYYMMDD-HHMMSS-ralph/)
These contain session-specific archives from Ralph automated development runs, including temporary files and intermediate states.

## Why Archived?

The cloud API approach had several limitations:
1. **Cost** - Per-API-call pricing becomes expensive at scale
2. **Latency** - Round-trip API calls add significant delays
3. **Quality** - Local Qwen3-32B provides better context-aware translations
4. **Control** - Self-hosted infrastructure allows full customization

## Migration Path

The new architecture uses:
- **RunPod Serverless** - GPU workers for ML inference
- **Qwen3-32B LLM** - Local translation with better quality
- **Whisper (local)** - Faster transcription without API calls
- **Supabase Storage** - Result file hosting

See the main `README.md` and `runpod/` directory for the current implementation.
