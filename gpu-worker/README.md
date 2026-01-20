# Self-hosted GPU Multi-Container Deployment

This directory contains the 5-worker architecture for video translation:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Self-hosted GPU Serverless                                │
│                                                                         │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐  │
│ │ Whisper     │ │ Pyannote    │ │ Demucs  │ │ vLLM    │ │Chatterbox │  │
│ │ (PRE-BUILT) │ │ (CUSTOM)    │ │(CUSTOM) │ │(PRE-BLT)│ │ (CUSTOM)  │  │
│ │ Official    │ │ Diarization │ │ Audio   │ │ Qwen3   │ │ TTS       │  │
│ └─────────────┘ └─────────────┘ └─────────┘ └─────────┘ └───────────┘  │
│   ~8GB           ~3GB            ~5GB        Already ✅    ~8GB          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Workers

| Worker | Type | GPU | Description |
|--------|------|-----|-------------|
| **Faster-Whisper** | Pre-built | RTX 4090 | Speech-to-text transcription |
| **Pyannote Diarization** | Custom | RTX 4090 | Speaker identification |
| **Demucs** | Custom | RTX 4090 | Audio source separation (vocals/background) |
| **vLLM (Qwen3-32B)** | Pre-built | A100 80GB | Translation with context |
| **Chatterbox TTS** | Custom | RTX 4090 | Voice cloning text-to-speech |

## Directory Structure

```
self-hosted/
├── diarization/          # Pyannote speaker diarization worker
│   ├── Dockerfile
│   ├── handler.py
│   └── requirements.txt
├── demucs/               # Demucs audio separation worker
│   ├── Dockerfile
│   ├── handler.py
│   └── requirements.txt
├── chatterbox/           # Chatterbox TTS worker
│   ├── Dockerfile
│   ├── handler.py
│   └── requirements.txt
├── vllm/                 # vLLM client and docs
│   ├── client.py
│   └── README.md
├── orchestrator.py       # Pipeline coordinator
└── README.md             # This file
```

## Current Endpoint IDs

```env
# Pre-built Workers
GPU_PROVIDER_WHISPER_ENDPOINT_ID=       # Deploy from Self-hosted GPU Hub
GPU_PROVIDER_VLLM_ENDPOINT_ID=mo5vt1mlci0mtw  # ✅ Working

# Custom Workers (deploy after GitHub Actions builds images)
GPU_PROVIDER_DIARIZATION_ENDPOINT_ID=
GPU_PROVIDER_DEMUCS_ENDPOINT_ID=
GPU_PROVIDER_CHATTERBOX_ENDPOINT_ID=
```

## Deployment

### Step 1: Images are built by GitHub Actions

Images are automatically built and pushed to GHCR when code changes:
- `ghcr.io/fjsikora/yt-translate/diarization:latest`
- `ghcr.io/fjsikora/yt-translate/demucs:latest`
- `ghcr.io/fjsikora/yt-translate/chatterbox:latest`

Check build status: https://github.com/fjsikora/yt-translate/actions

### Step 2: Deploy Pre-built Whisper

1. Go to Self-hosted GPU Console → Serverless → Explore
2. Search "Faster Whisper"
3. Deploy with:
   - GPU: RTX 4090
   - Workers: 0-3

### Step 3: Create Templates for Custom Workers

Using Self-hosted GPU MCP or dashboard, create templates:

**Diarization Template:**
```
Name: yt-diarization
Image: ghcr.io/fjsikora/yt-translate/diarization:latest
Container Disk: 20GB
Volume: /workspace
Env: HUGGINGFACE_TOKEN
```

**Demucs Template:**
```
Name: yt-demucs
Image: ghcr.io/fjsikora/yt-translate/demucs:latest
Container Disk: 20GB
Volume: /workspace
```

**Chatterbox Template:**
```
Name: yt-chatterbox
Image: ghcr.io/fjsikora/yt-translate/chatterbox:latest
Container Disk: 30GB
Volume: /workspace
```

### Step 4: Create Endpoints

For each template, create a serverless endpoint:
- GPU: RTX 4090
- Min workers: 0
- Max workers: 3
- Flash Boot: Enabled

## Environment Variables

### Self-hosted API Gateway
```env
GPU_PROVIDER_API_KEY=rp_xxxxx
GPU_PROVIDER_WHISPER_ENDPOINT_ID=xxxxx
GPU_PROVIDER_DIARIZATION_ENDPOINT_ID=xxxxx
GPU_PROVIDER_DEMUCS_ENDPOINT_ID=xxxxx
GPU_PROVIDER_VLLM_ENDPOINT_ID=mo5vt1mlci0mtw
GPU_PROVIDER_CHATTERBOX_ENDPOINT_ID=xxxxx
HUGGINGFACE_TOKEN=hf_xxxxx
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=xxxxx
```

### Worker-specific
```env
# Diarization worker
HUGGINGFACE_TOKEN=hf_xxxxx  # Required for Pyannote

# vLLM worker
MODEL_NAME=Qwen/Qwen3-32B
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.95
```

## Pipeline Flow

```
Video URL
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
┌────────┐   ┌──────────┐   ┌─────────┐
│Whisper │   │Diarization│  │ Demucs  │
│  STT   │   │ (parallel)│  │(parallel)│
└───┬────┘   └────┬──────┘  └────┬────┘
    │             │              │
    ▼             ▼              ▼
 Segments    Speaker IDs    Vocals + BG
    │             │              │
    └──────┬──────┴──────────────┘
           ▼
    ┌─────────────┐
    │ vLLM Qwen3  │
    │ Translation │
    └──────┬──────┘
           ▼
    Translated Text
           │
           ▼
    ┌─────────────┐
    │ Chatterbox  │
    │    TTS      │
    └──────┬──────┘
           ▼
    Dubbed Audio + Mix
```

## Cost Estimate

| Worker | GPU | Est. Time | Cost/run |
|--------|-----|-----------|----------|
| Whisper | RTX 4090 | ~30s | $0.009 |
| Diarization | RTX 4090 | ~30s | $0.009 |
| Demucs | RTX 4090 | ~30s | $0.009 |
| vLLM | A100 80GB | ~60s | $0.083 |
| Chatterbox | RTX 4090 | ~120s | $0.037 |
| **Total** | | ~270s | **~$0.15/video** |

At 50 videos/day = **~$225/mo**

## Testing Endpoints

### Test Diarization
```bash
curl -X POST "https://api.self-hosted.ai/v2/${GPU_PROVIDER_DIARIZATION_ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${GPU_PROVIDER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/audio.wav"
    }
  }'
```

### Test Demucs
```bash
curl -X POST "https://api.self-hosted.ai/v2/${GPU_PROVIDER_DEMUCS_ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${GPU_PROVIDER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/audio.wav"
    }
  }'
```

### Test Chatterbox
```bash
curl -X POST "https://api.self-hosted.ai/v2/${GPU_PROVIDER_CHATTERBOX_ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${GPU_PROVIDER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "segments": [{"id": 0, "text": "Hello world", "speaker": "SPEAKER_00"}],
      "voice_samples": {"SPEAKER_00": "https://example.com/voice.wav"}
    }
  }'
```
