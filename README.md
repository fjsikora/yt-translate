# yt-translate

An open source video translation and dubbing studio. Paste a video URL, pick a target language, and get a dubbed video with cloned speaker voices.

**Self-hosted pipeline running on [Self-hosted GPU](https://your-gpu-provider).** No per-minute API costs for transcription or TTS — you pay only for the GPU time you use.

---

## Features

- **Transcription** — OpenAI Whisper large-v3 for accurate multilingual transcription
- **Speaker diarization** — pyannote.audio 3.1 to identify and separate speakers
- **Voice cloning** — Chatterbox TTS (ResembleAI) clones each speaker's voice in the target language
- **Background separation** — Demucs htdemucs separates vocals from music/ambience for clean mixing
- **Translation** — Deep Translator (Google Translate) with optional local LLM refinement
- **Timeline editor** — Next.js studio for reviewing and editing dubbed segments before export
- **23 languages** — Arabic, Chinese, Danish, Dutch, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish, and English

---

## Architecture

```
Video URL
    │
    ▼
yt-dlp (download)
    │
    ├─► Whisper (transcription + timestamps)
    │
    ├─► pyannote.audio (speaker diarization)
    │
    ├─► Demucs (vocal / background separation)
    │
    ├─► Deep Translator / LLM (translation)
    │
    ├─► Chatterbox TTS (voice cloning per speaker)
    │
    └─► ffmpeg (mix + sync → dubbed video)

Frontend: Next.js dubbing studio (timeline editor, segment review, export)
Backend:  FastAPI + Self-hosted GPU serverless workers
```

---

## Prerequisites

- [Self-hosted GPU](https://your-gpu-provider) account (GPU compute)
- [Supabase](https://supabase.com) project (database + storage)
- [HuggingFace](https://huggingface.co) account with access to:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- ffmpeg installed locally (for development)

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/fjsikora/yt-translate.git
cd yt-translate
cp .env.example .env
# Fill in .env with your credentials (see below)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API server

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run the studio (frontend)

```bash
cd studio
npm install
npm run dev
# Open http://localhost:3000
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | Yes | Your Supabase project URL |
| `SUPABASE_PROJECT_REF` | Yes | Your Supabase project reference ID |
| `SUPABASE_SERVICE_KEY` | Yes | Supabase service role key |
| `HUGGINGFACE_TOKEN` | Yes | HuggingFace token (pyannote model access) |
| `GPU_PROVIDER_API_KEY` | Yes | Self-hosted GPU API key |
| `DUBBING_STUDIO_POD_URL` | Yes | Self-hosted GPU pod URL for AI processing |
| `GPU_PROVIDER_TEMPLATE_ID` | No | Self-hosted GPU pod template ID |
| `REPLICATE_API_TOKEN` | No | Replicate token (alternative TTS/LLM backend) |
| `OXYLABS_PROXY` | No | Proxy for high-volume YouTube downloads |
| `YOUTUBE_COOKIES` | No | Netscape-format cookies (bypass bot detection) |

See `.env.example` for the full list with comments.

---

## Self-hosted GPU Deployment

The pipeline runs as a unified worker on Self-hosted GPU. A single pod handles all stages: Whisper, Demucs, pyannote, Chatterbox TTS, and optional local LLM translation.

```bash
# Build and push the unified worker image
cd self-hosted
docker build -f Dockerfile.unified -t your-registry/yt-translate-worker:latest .
docker push your-registry/yt-translate-worker:latest
```

Then create a Self-hosted GPU template pointing to your image and set `DUBBING_STUDIO_POD_URL` in your `.env`.

---

## Supported Languages

Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish

---

## Tech Stack

| Layer | Tech |
|---|---|
| Transcription | OpenAI Whisper large-v3 |
| Speaker diarization | pyannote.audio 3.1 |
| Source separation | Demucs htdemucs |
| TTS / voice cloning | Chatterbox TTS (ResembleAI) |
| Translation | Deep Translator + optional local LLM |
| Video processing | yt-dlp + ffmpeg |
| Backend | FastAPI (Python) |
| Frontend | Next.js + Tailwind |
| Database + Storage | Supabase |
| GPU compute | Self-hosted GPU |

---

## Project Structure

```
yt-translate/
├── yt_translate/
│   ├── core/           # Translation pipeline orchestration
│   ├── processing/     # Audio, LLM, lipsync processing
│   ├── workers/        # Self-hosted GPU worker handlers (Whisper, Demucs, TTS, etc.)
│   ├── api/            # FastAPI server + Self-hosted GPU client
│   └── config/         # Constants and language mappings
├── studio/             # Next.js dubbing studio frontend
├── self-hosted/             # Unified Self-hosted GPU worker Dockerfile
├── scripts/            # Database migration utilities
└── tests/              # Test suite
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

AI models used:
- **Whisper** — MIT License (OpenAI)
- **Chatterbox TTS** — MIT License (ResembleAI, includes perceptual watermarking)
- **Demucs** — MIT License (Meta)

---

## Contributing

Pull requests welcome. For major changes, open an issue first to discuss what you'd like to change.
