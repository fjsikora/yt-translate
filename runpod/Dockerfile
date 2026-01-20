# RunPod Serverless Dockerfile for Video Translation Worker
#
# This image packages all ML models and dependencies for RunPod deployment.
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
COPY requirements.txt /app/requirements.txt

# Install Python dependencies from main requirements.txt
# Note: Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install additional RunPod-specific dependencies
# (transformers, accelerate, bitsandbytes for Qwen3-32B LLM translation)
# (runpod SDK for serverless handler)
# (supabase for storage upload)
RUN pip install --no-cache-dir \
    runpod \
    supabase \
    transformers \
    accelerate \
    bitsandbytes \
    scipy \
    stripe

# Pre-download Whisper medium model
# The model will be cached in ~/.cache/whisper
RUN python -c "import whisper; whisper.load_model('medium')"

# Pre-download Demucs htdemucs model
# The model will be cached in ~/.cache/torch/hub
RUN python -c "import torch; torch.hub.load('facebookresearch/demucs', 'htdemucs', source='github')"

# Copy application code
COPY *.py /app/
COPY runpod/ /app/runpod/

# Set working directory to app root
WORKDIR /app

# Run the handler as entrypoint
CMD ["python", "runpod/handler.py"]
