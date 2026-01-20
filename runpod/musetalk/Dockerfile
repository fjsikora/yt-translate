# MuseTalk Lip Sync Worker Dockerfile
# Complex dependencies: OpenMMLab (mmcv, mmdet, mmpose) + MuseTalk
# Estimated image size: ~15GB

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.cache/torch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default (MuseTalk requires Python 3.10)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Install PyTorch first
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install OpenMMLab packages via mim
RUN pip install --no-cache-dir openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.1" && \
    mim install "mmdet>=3.1.0" && \
    mim install "mmpose>=1.1.0"

# Clone MuseTalk repository
RUN git clone https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk

# Install MuseTalk dependencies
WORKDIR /app/MuseTalk
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install handler requirements
COPY runpod/musetalk/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Download MuseTalk models at build time for faster cold starts
# Note: These are large downloads, comment out for smaller image
RUN mkdir -p /app/MuseTalk/models/musetalk && \
    mkdir -p /app/MuseTalk/models/dwpose && \
    mkdir -p /app/MuseTalk/models/face-parse-bisent && \
    mkdir -p /app/MuseTalk/models/sd-vae-ft-mse && \
    mkdir -p /app/MuseTalk/models/whisper

# Download models using huggingface-cli or wget
# MuseTalk model
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('TMElyralab/MuseTalk', local_dir='/app/MuseTalk/models/musetalk', local_dir_use_symlinks=False)" || true

# Copy handler
COPY runpod/musetalk/handler.py /app/handler.py

# Create cache directory
RUN mkdir -p /workspace/.cache/torch

WORKDIR /app

# Set entrypoint
CMD ["python", "/app/handler.py"]
