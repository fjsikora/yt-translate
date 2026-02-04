# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered video dubbing platform: upload a video, select source/target languages, and get a dubbed version with voice-cloned speech.

- **Frontend**: Next.js 15 App Router (`studio/`) — timeline-based dubbing editor with multi-track audio
- **Backend**: FastAPI on Self-hosted GPU GPU (`self-hosted/handler_unified.py`) — single unified image with all ML models
- **Database/Storage**: Supabase (PostgreSQL + object storage)

## Common Commands

### Frontend (studio/)
```bash
cd studio
npm install
npm run dev          # http://localhost:3000
npx tsc --noEmit     # type-check
```

### Backend (local GPU)
```bash
docker compose up --build   # runs on port 8000
curl http://localhost:8000/health
```

### CI/CD
Docker images are built via GitHub Actions and pushed to GHCR (`ghcr.io/{owner}/yt-translate/dubbing-studio`).

- `build-dubbing-studio.yml` — triggers on push to `main` when `self-hosted/handler_unified.py`, `self-hosted/Dockerfile.unified`, or `self-hosted/requirements.unified.txt` change
- `build-self-hosted-workers.yml` — triggers for individual worker images (demucs, whisper, chatterbox, etc.)

After pushing pipeline changes, wait for the GHCR build (~30 min), then restart the Self-hosted GPU pod to pull the new image.

## Architecture

### Dubbing Pipeline (backend)
`self-hosted/handler_unified.py` orchestrates the full pipeline via `POST /dub`:

1. **Extract audio** (ffmpeg) from uploaded video
2. **Transcribe** (Whisper large-v3) → timestamped segments
3. **Diarize** (pyannote 3.1) → speaker labels
4. **Separate** (Demucs htdemucs) → vocals + background
5. **Translate** (Qwen3-8B GGUF via llama-cpp) → target language text
6. **Synthesize** (Chatterbox Multilingual TTS) → voice-cloned speech per segment
7. **Mix** (ffmpeg) → dubbed speech + background

Results are saved to Supabase: 4 tracks (video, background, vocals, dubbed) with segments. The `tts_duration` field captures actual TTS audio length; dubbed segments use shift-based layout so they play at natural 1x speed without trimming or overlap.

### Frontend Data Flow
- **Project CRUD** and **track/segment updates** go directly to Supabase via `@supabase/supabase-js` client (`studio/src/lib/supabase.ts`)
- **Pipeline trigger** (`POST /dub`) and **export** (`POST /api/projects/{id}/export`) go to Self-hosted GPU via `NEXT_PUBLIC_API_URL`
- **Audio playback** uses Web Audio API (`studio/src/lib/audio-engine.ts`) — preloads buffers, schedules in 30s windows, applies per-segment `speedFactor` via `playbackRate`
- **Video playback** syncs with timeline via `playbackRate` prop on VideoPlayer, converting between video-time and timeline-time

### Key Frontend Files
| File | Role |
|------|------|
| `studio/src/app/projects/[id]/edit/page.tsx` | Main editor — fetches project+tracks from Supabase, manages playback state, segment CRUD |
| `studio/src/components/editor/Timeline.tsx` | Timeline with drag/trim/stretch (Alt+drag for speed 0.5-2x), track controls |
| `studio/src/components/editor/VideoPlayer.tsx` | HTML5 video with playbackRate support and keyboard shortcuts |
| `studio/src/lib/audio-engine.ts` | Web Audio API engine — multi-track scheduling with speed factor |
| `studio/src/stores/timeline-store.ts` | Zustand store — tracks, playback, zoom, undo/redo (50-item history) |

### Database Schema (Supabase)
Migrations in `supabase_migrations/migrations/`.

- **dub_projects** — id, name, status (`pending|processing|ready|exporting|exported|error`), languages, `*_url` fields, duration
- **dub_tracks** — project_id, name, type (`video|background|vocals|dubbed`), muted/solo/volume, order_index
- **dub_segments** — track_id, speaker, original_text, translated_text, start_time, end_time, speed_factor, audio_url

Storage buckets: `dub-videos`, `dub-audio`, `dub-exports`. RLS policies currently allow anonymous access (no auth system yet).

### Track Types
| Type | Color | Purpose |
|------|-------|---------|
| `video` | amber | Controls video playback speed (no audio) |
| `background` | green | Instrumental/ambient audio (demucs output) |
| `vocals` | blue | Original speech (muted by default, for reference) |
| `dubbed` | purple | TTS-generated translated speech segments |

## Environment Variables

### Frontend (`studio/.env.local`)
- `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` — Supabase connection
- `NEXT_PUBLIC_API_URL` — Self-hosted GPU pod URL (for `/dub` and `/export` endpoints)

### Backend (Self-hosted GPU env or `.env`)
- `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` — Supabase service-role access
- `HUGGINGFACE_TOKEN` — for pyannote model download at build time
