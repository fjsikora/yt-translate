"""
Self-hosted GPU llama.cpp Worker Handler

Handles LLM inference using llama-cpp-python for GGUF models.
Provides OpenAI-compatible chat completions API.
"""

import os
import time
import traceback
from pathlib import Path

import self-hosted


def log(msg: str) -> None:
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_startup_info() -> None:
    """Log startup health check information."""
    log("=== llama.cpp Worker Starting ===")

    # Check for CUDA
    try:
        import torch
        log(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                log(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        log("PyTorch not installed (not required for llama.cpp)")

    # Check llama-cpp-python
    try:
        import llama_cpp
        log(f"llama-cpp-python version: {llama_cpp.__version__}")
    except ImportError as e:
        log(f"ERROR: llama-cpp-python not installed: {e}")

    # Log configuration
    model_repo = os.environ.get("MODEL_REPO", "unsloth/Qwen3-32B-GGUF")
    model_file = os.environ.get("MODEL_FILE", "Qwen3-32B-Q4_1.gguf")
    model_path = os.environ.get("MODEL_PATH", f"/models/{model_file}")
    log(f"Model repo: {model_repo}")
    log(f"Model file: {model_file}")
    log(f"Model path: {model_path}")
    log(f"Model exists: {Path(model_path).exists()}")

    log("=== Startup Complete ===")


# Global model for reuse
_llm = None


def download_model() -> str:
    """Download GGUF model from HuggingFace if not present."""
    from huggingface_hub import hf_hub_download

    model_repo = os.environ.get("MODEL_REPO", "unsloth/Qwen3-32B-GGUF")
    model_file = os.environ.get("MODEL_FILE", "Qwen3-32B-Q4_1.gguf")
    model_path = os.environ.get("MODEL_PATH", f"/models/{model_file}")

    if Path(model_path).exists():
        log(f"Model already exists: {model_path}")
        return model_path

    log(f"Downloading {model_repo}/{model_file}...")
    log("(This may take several minutes for large models)")

    download_start = time.time()
    local_path = hf_hub_download(
        repo_id=model_repo,
        filename=model_file,
        local_dir="/models",
    )
    download_time = time.time() - download_start
    log(f"Model downloaded in {download_time:.1f}s: {local_path}")

    return model_path


def get_llm():
    """Get or create the llama.cpp model."""
    global _llm

    if _llm is None:
        from llama_cpp import Llama

        model_path = download_model()
        n_ctx = int(os.environ.get("N_CTX", "8192"))
        n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", "-1"))  # -1 = all layers

        log(f"Loading model from {model_path}...")
        log(f"Context length: {n_ctx}")
        log(f"GPU layers: {n_gpu_layers}")
        log("(This may take a minute...)")

        load_start = time.time()
        _llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
        load_time = time.time() - load_start
        log(f"Model loaded successfully in {load_time:.1f}s")

    return _llm


def validate_input(job_input: dict) -> None:
    """Validate job input."""
    if "messages" not in job_input:
        raise ValueError("'messages' is required")

    messages = job_input["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("'messages' must be a non-empty list")


def handler(job: dict) -> dict:
    """Main handler for llama.cpp jobs."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    start_time = time.time()

    try:
        log(f"Job {job_id} started")
        log(f"Input keys: {list(job_input.keys())}")

        validate_input(job_input)

        # Get parameters
        messages = job_input["messages"]
        max_tokens = job_input.get("max_tokens", 1024)
        temperature = job_input.get("temperature", 0.7)
        top_p = job_input.get("top_p", 0.9)

        # Get model
        llm = get_llm()

        log(f"Running inference (max_tokens={max_tokens}, temp={temperature})...")
        inference_start = time.time()

        # Generate response
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        inference_time = time.time() - inference_start
        log(f"Inference completed in {inference_time:.1f}s")

        # Extract response text
        response_text = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})

        total_time = round(time.time() - start_time, 2)
        log(f"Job {job_id} completed in {total_time}s")

        return {
            "status": "success",
            "response": response_text,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "metrics": {
                "inference_seconds": round(inference_time, 2),
                "total_seconds": total_time,
            },
        }

    except ValueError as e:
        log(f"Validation error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": "validation",
        }
    except Exception as e:
        log(f"Error: {e}")
        log(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "error_type": "runtime",
            "error_trace": traceback.format_exc(),
        }


if __name__ == "__main__":
    log_startup_info()
    self-hosted.serverless.start({
        "handler": handler,
        "refresh_worker": False,  # Keep model loaded between jobs
    })
