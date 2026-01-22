# llama.cpp Worker Dockerfile
# Uses llama-cpp-python for GGUF model inference
# Model downloaded at runtime from HuggingFace

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_REPO=unsloth/Qwen3-32B-GGUF
ENV MODEL_FILE=Qwen3-32B-Q4_1.gguf
ENV MODEL_PATH=/models/Qwen3-32B-Q4_1.gguf

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
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

# Install pre-built llama-cpp-python with CUDA support (avoids compilation)
RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Install other dependencies
RUN pip install --no-cache-dir \
    self-hosted>=1.7.0 \
    huggingface_hub>=0.20.0

# Copy handler
COPY self-hosted/vllm/handler.py /app/handler.py

# Create models directory
RUN mkdir -p /models

# Model downloaded at runtime (first request)
# ~18GB GGUF file from unsloth/Qwen3-32B-GGUF

# Set entrypoint
CMD ["python", "/app/handler.py"]
