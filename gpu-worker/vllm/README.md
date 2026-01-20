# vLLM Endpoint for Qwen3-32B Translation

This endpoint uses Self-hosted GPU's pre-built vLLM Worker template for optimized LLM inference.

## Deployed Endpoint

**Endpoint ID:** `mo5vt1mlci0mtw`
**Endpoint Name:** `yt-vllm`
**Template ID:** `gxd9dfcy8l`
**GPU:** NVIDIA A100 80GB PCIe
**Workers:** Min 0, Max 1 (scale to zero)
**Idle Timeout:** 10 seconds

### Configuration

The endpoint uses these environment variables (configured in template `gxd9dfcy8l`):

```
MODEL_NAME=Qwen/Qwen3-32B
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.95
TENSOR_PARALLEL_SIZE=1
HF_TOKEN=<set in Self-hosted GPU dashboard>
```

### OpenAI-Compatible API

The vLLM Worker provides an OpenAI-compatible API:

**Base URL:** `https://api.self-hosted.ai/v2/mo5vt1mlci0mtw/openai/v1`

Supported endpoints:
- `POST /v1/chat/completions` (streaming and non-streaming)
- `GET /v1/models`

## Test the Endpoint

```bash
curl -X POST "https://api.self-hosted.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${GPU_PROVIDER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [
        {"role": "system", "content": "You are a translator."},
        {"role": "user", "content": "Translate to Spanish: Hello world"}
      ],
      "max_tokens": 100,
      "temperature": 0.3
    }
  }'
```

## API Format

### Input
```json
{
  "input": {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ],
    "max_tokens": 4096,
    "temperature": 0.3
  }
}
```

### Output
```json
{
  "output": {
    "choices": [
      {
        "message": {
          "content": "translated text..."
        }
      }
    ]
  }
}
```

## Environment Variables

Add to Self-hosted API gateway:
```
GPU_PROVIDER_VLLM_ENDPOINT_ID=mo5vt1mlci0mtw
```

## Cost Estimate

- A100 80GB: ~$0.00076/sec = $2.74/hr
- RTX 4090: ~$0.00031/sec = $1.12/hr (with quantization)

For a 5-minute video with ~50 segments:
- Translation time: ~60-120 seconds
- Cost: ~$0.05-0.10 per video
