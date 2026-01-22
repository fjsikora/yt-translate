# vLLM Worker Dockerfile
# Based on Self-hosted GPU's vLLM worker image for compatibility
# Model downloaded at runtime and cached in volume
# Requires A100 80GB for Qwen3-32B

FROM self-hosted/worker-v1-vllm:v2.11.2-cuda12.4.1

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir self-hosted>=1.7.0

# Copy handler
COPY self-hosted/vllm/handler.py /app/handler.py

# Environment variables for model (can be overridden at runtime)
ENV MODEL_NAME="Qwen/Qwen3-32B"
ENV MAX_MODEL_LEN="8192"
ENV GPU_MEMORY_UTILIZATION="0.95"

# Set entrypoint to our custom handler
CMD ["python", "/app/handler.py"]
