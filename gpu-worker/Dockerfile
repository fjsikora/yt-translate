# Self-hosted GPU Serverless Dockerfile for Video Translation Worker
#
# This image packages all ML models and dependencies for Self-hosted GPU deployment.
# Models are pre-downloaded and baked into the image for faster cold starts.

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python to not buffer output (for better logging)
ENV PYTHONUNBUFFERED=1

# Set cache directories for models
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY self-hosted/requirements.txt /app/self-hosted/requirements.txt

# Install Python dependencies from dedicated self-hosted requirements
# All versions are pinned for reproducibility
# Note: Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r /app/self-hosted/requirements.txt

# Pre-download Whisper medium model
# The model will be cached in ~/.cache/whisper
RUN python -c "import whisper; whisper.load_model('medium')"

# Pre-download Demucs htdemucs model
# The model will be cached in ~/.cache/torch/hub
RUN python -c "import torch; torch.hub.load('facebookresearch/demucs', 'htdemucs', source='github')"

# Copy application code
COPY *.py /app/
COPY self-hosted/ /app/self-hosted/

# Set working directory to app root
WORKDIR /app

# Run the handler as entrypoint
CMD ["python", "self-hosted/handler.py"]
