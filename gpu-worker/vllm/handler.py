"""
Self-hosted GPU vLLM Worker Handler

Handles LLM inference using vLLM for Qwen3-32B.
Provides OpenAI-compatible chat completions API.
"""

import os
import sys
import time
import traceback

import self-hosted
import torch


def log(msg: str) -> None:
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_startup_info() -> None:
    """Log startup health check information."""
    log("=== vLLM Worker Starting ===")
    log(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            log(f"  Memory: {mem_total:.1f} GB")
        log(f"CUDA Version: {torch.version.cuda}")

    # Check vLLM
    try:
        import vllm
        log(f"vLLM version: {vllm.__version__}")
    except ImportError as e:
        log(f"ERROR: vllm not installed: {e}")

    # Log configuration
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-32B")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "8192")
    gpu_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.95")
    log(f"Model: {model_name}")
    log(f"Max model length: {max_model_len}")
    log(f"GPU memory utilization: {gpu_utilization}")

    log("=== Startup Complete ===")


# Global model for reuse
_llm = None


def get_llm():
    """Get or create the vLLM engine."""
    global _llm

    if _llm is None:
        from vllm import LLM

        model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-32B")
        max_model_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
        gpu_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))
        tensor_parallel = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        log(f"Loading model {model_name}...")
        log("(This may take several minutes on first load)")

        _llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_utilization,
            tensor_parallel_size=tensor_parallel,
            trust_remote_code=True,
            dtype="auto",
        )

        log("Model loaded successfully")

    return _llm


def format_messages(messages: list) -> str:
    """Format messages for chat completion."""
    # Simple chat template for Qwen3
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


def run_inference(
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    """Run LLM inference."""
    from vllm import SamplingParams

    llm = get_llm()

    # Format prompt
    prompt = format_messages(messages)

    log(f"Running inference (max_tokens={max_tokens}, temp={temperature})...")
    inference_start = time.time()

    # Create sampling params
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # Generate
    outputs = llm.generate([prompt], sampling_params)

    inference_time = time.time() - inference_start
    log(f"Inference completed in {inference_time:.1f}s")

    # Extract response
    output = outputs[0]
    response_text = output.outputs[0].text.strip()

    # Calculate token counts
    prompt_tokens = len(output.prompt_token_ids)
    completion_tokens = len(output.outputs[0].token_ids)

    return {
        "response": response_text,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "inference_time": inference_time,
    }


def validate_input(job_input: dict) -> None:
    """Validate job input."""
    if "messages" not in job_input:
        raise ValueError("'messages' is required")

    messages = job_input["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("'messages' must be a non-empty list")


def handler(job: dict) -> dict:
    """Main handler for vLLM jobs."""
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

        # Run inference
        result = run_inference(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        total_time = round(time.time() - start_time, 2)
        log(f"Job {job_id} completed in {total_time}s")

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "response": result["response"],
            "usage": result["usage"],
            "metrics": {
                "inference_seconds": round(result["inference_time"], 2),
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
