# MuseTalk Lip Sync Worker Dockerfile
# Complex dependencies: OpenMMLab (mmcv, mmdet, mmpose) + MuseTalk
# Models downloaded at BUILD TIME for fast cold starts
# Estimated image size: ~15-18GB

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.cache/torch
ENV HF_HOME=/workspace/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default (MuseTalk requires Python 3.10)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Install PyTorch 2.1.0 with CUDA 12.1 (compatible with CUDA 12.4 runtime)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install OpenMMLab packages via mim (avoids version conflicts)
# Note: chumpy has a broken setup.py, install it separately first
RUN pip install --no-cache-dir openmim && \
    pip install --no-cache-dir --no-build-isolation chumpy && \
    mim install mmengine && \
    mim install "mmcv>=2.0.1,<2.2.0" && \
    mim install "mmdet>=3.1.0" && \
    mim install "mmpose>=1.1.0"

# Clone MuseTalk repository
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk

# Install MuseTalk dependencies
WORKDIR /app/MuseTalk
RUN pip install --no-cache-dir \
    diffusers==0.30.2 \
    accelerate==0.28.0 \
    numpy==1.23.5 \
    opencv-python==4.9.0.80 \
    soundfile==0.12.1 \
    transformers==4.39.2 \
    huggingface_hub>=0.20.0 \
    librosa==0.11.0 \
    einops==0.8.1 \
    omegaconf \
    imageio[ffmpeg] \
    ffmpeg-python \
    moviepy \
    gdown \
    requests

# Copy and install handler requirements
COPY runpod/musetalk/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create model directories
RUN mkdir -p /app/MuseTalk/models/musetalk \
    /app/MuseTalk/models/sd-vae-ft-mse \
    /app/MuseTalk/models/whisper \
    /app/MuseTalk/models/dwpose \
    /app/MuseTalk/models/face-parse-bisent

# Download models at build time for fast cold starts
# MuseTalk models from HuggingFace
RUN python -c "\
from huggingface_hub import snapshot_download; \
print('Downloading MuseTalk models...'); \
snapshot_download('TMElyralab/MuseTalk', local_dir='/app/MuseTalk/models', local_dir_use_symlinks=False, allow_patterns=['*.json', '*.bin', '*.pth', '*.pt', '*.safetensors']); \
print('Downloading SD VAE...'); \
snapshot_download('stabilityai/sd-vae-ft-mse', local_dir='/app/MuseTalk/models/sd-vae-ft-mse', local_dir_use_symlinks=False); \
print('Downloading Whisper...'); \
snapshot_download('openai/whisper-tiny', local_dir='/app/MuseTalk/models/whisper', local_dir_use_symlinks=False); \
print('Downloading DWPose...'); \
snapshot_download('yzd-v/DWPose', local_dir='/app/MuseTalk/models/dwpose', local_dir_use_symlinks=False); \
print('Model downloads complete!'); \
"

# Create symlink for MuseTalk's expected path (expects 'sd-vae' not 'sd-vae-ft-mse')
RUN ln -s /app/MuseTalk/models/sd-vae-ft-mse /app/MuseTalk/models/sd-vae

# Create symlink for config.json (inference.py expects config.json but HF has musetalk.json)
RUN ln -sf /app/MuseTalk/models/musetalk/musetalk.json /app/MuseTalk/models/musetalk/config.json

# Download face-parse-bisent models
RUN wget -q https://download.pytorch.org/models/resnet18-5c106cde.pth \
    -O /app/MuseTalk/models/face-parse-bisent/resnet18-5c106cde.pth

# Download 79999_iter.pth from Google Drive using gdown
RUN gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
    -O /app/MuseTalk/models/face-parse-bisent/79999_iter.pth || \
    echo "Warning: gdown failed, model may need manual download"

# Copy handler
COPY runpod/musetalk/handler.py /app/handler.py

# Create cache directories
RUN mkdir -p /workspace/.cache/torch /workspace/.cache/huggingface

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
RUN python -c "import mmcv; import mmdet; import mmpose; print('OpenMMLab packages OK')"

WORKDIR /app

# Set entrypoint
CMD ["python", "/app/handler.py"]
