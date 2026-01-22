# Diarization Worker Dockerfile
# pyannote.audio speaker diarization
# Model pre-downloaded for faster cold starts
# Estimated image size: ~8GB

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.cache/torch

# HuggingFace token for model download (set at build time or runtime)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Install PyTorch 2.5.x (avoid 2.6 compatibility issues)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy and install requirements
COPY self-hosted/diarization/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download pyannote model for faster cold starts
# Note: Requires HF_TOKEN with access to pyannote models
RUN python -c "\
from pyannote.audio import Pipeline; \
import os; \
token = os.environ.get('HF_TOKEN', ''); \
if token: Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=token); \
else: print('Warning: HF_TOKEN not set, model will be downloaded at runtime')"

# Copy handler
COPY self-hosted/diarization/handler.py /app/handler.py

# Create cache directory
RUN mkdir -p /workspace/.cache/torch

# Set entrypoint
CMD ["python", "/app/handler.py"]
