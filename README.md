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

## Requirements

### GPU

| Component | VRAM |
|---|---|
| Whisper large-v3 | ~2GB |
| Demucs htdemucs | ~3GB |
| pyannote.audio 3.1 | ~2GB |
| Chatterbox TTS | ~4GB |
| **Total (all models loaded)** | **~12GB minimum, 16GB recommended** |

A 16GB GPU (e.g. RTX 3080/4080, RTX 4000 Ada, A4000) runs the full pipeline comfortably. CPU-only is possible but very slow for anything beyond short clips.

### System

- Python 3.10+
- ffmpeg (must be installed and on PATH)
- CUDA 11.8+ (for GPU acceleration)

---

## Quick Start

**1. Clone and configure**

```bash
git clone https://github.com/fjsikora/yt-translate.git
cd yt-translate
pip install -r requirements.txt
cp .env.example .env
```

Set at minimum in your `.env`:
- `HUGGINGFACE_TOKEN` — required for pyannote speaker diarization models
- `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` + `SUPABASE_PROJECT_REF` — for job persistence and file storage

**2. Run the API server**

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

**3. Run the studio (frontend)**

```bash
cd studio
npm install
cp .env.example .env.local
# Set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
# Open http://localhost:3000
```

**4. Or use the CLI directly**

```bash
python translate.py
# Interactive prompts: paste a video URL, pick a target language
# Output saved to output/
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
