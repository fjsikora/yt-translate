#!/usr/bin/env python3
"""
Test script for the full translation pipeline.
Tests US-018: End-to-End Testing - Local
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env file FIRST
load_dotenv(Path(__file__).parent / ".env")

# Force GPU 1 (RTX 5060 Ti) before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(f"HUGGINGFACE_TOKEN: {'set' if os.environ.get('HUGGINGFACE_TOKEN') else 'NOT SET'}")

import torch
import whisper
import yt_dlp
from deep_translator import GoogleTranslator
from audio_processing import AudioSeparator, SpeakerDiarizer

# Test configuration
TEST_URL = "https://www.youtube.com/watch?v=FN2RM-CHkuI"
TARGET_LANG = "Spanish"
LANG_CODE = "es"
OUTPUT_DIR = Path("output")

# Ensure output directory
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Test video: {TEST_URL}")
print(f"Target language: {TARGET_LANG}")
print()

# Import translate functions
from translate import (
    download_youtube, transcribe_audio, translate_segments,
    synthesize_segments, synthesize_segments_multi_speaker,
    mix_audio_with_background, generate_srt, merge_audio_video,
    get_iso_639_2_code, ProgressTracker
)


tracker = ProgressTracker(TEST_URL, TARGET_LANG)
def update(): pass

try:
    # Stage 1: Download (use existing files)
    print("[1/8] Downloading video...")
    video_path = OUTPUT_DIR / "FN2RM-CHkuI.mp4"
    audio_path = OUTPUT_DIR / "FN2RM-CHkuI.wav"
    title = "Exact Instructions Challenge PB&J Classroom Friendly | Josh Darnit"
    if video_path.exists() and audio_path.exists():
        print(f"  Using cached files")
    else:
        video_path, audio_path, title = download_youtube(TEST_URL, tracker, update)
    print(f"  Downloaded: {title}")
    print(f"  Video: {video_path}")
    print(f"  Audio: {audio_path}")

    # Stage 2: Separate audio (use cached if exists)
    vocals_path = OUTPUT_DIR / "vocals.wav"
    background_path = OUTPUT_DIR / "background.wav"
    if vocals_path.exists() and background_path.exists():
        print("[2/8] Using cached separated audio files...")
        print(f"  Vocals: {vocals_path}")
        print(f"  Background: {background_path}")
    else:
        print("[2/8] Separating vocals from background...")
        separator = AudioSeparator()
        vocals, background, sep_sample_rate = separator.separate(audio_path)
        vocals_path, background_path = separator.save_separated(
            vocals, background, sep_sample_rate, OUTPUT_DIR
        )
        print(f"  Vocals: {vocals_path}")
        print(f"  Background: {background_path}")

    # Stage 3: Speaker diarization
    print("[3/8] Identifying speakers...")
    diarizer = SpeakerDiarizer()
    diarization_segments = diarizer.diarize(vocals_path)
    speaker_samples = diarizer.extract_speaker_samples(
        vocals_path, OUTPUT_DIR, diarization_segments
    )
    num_speakers = len(speaker_samples)
    print(f"  Found {num_speakers} speaker(s): {list(speaker_samples.keys())}")

    # Stage 4: Transcribe
    print("[4/8] Transcribing audio...")
    segments = transcribe_audio(vocals_path, tracker, update)
    print(f"  {len(segments)} segments transcribed")

    total_duration = segments[-1]["end"] if segments else 0

    # Stage 5: Translate
    print(f"[5/8] Translating to {TARGET_LANG}...")
    translated_segments = translate_segments(segments, TARGET_LANG, tracker, update)
    print(f"  {len(translated_segments)} segments translated")

    # Stage 6: Synthesize
    print("[6/8] Synthesizing speech (multi-speaker)...")
    if num_speakers > 1 and speaker_samples:
        speech_audio = synthesize_segments_multi_speaker(
            translated_segments, diarization_segments, speaker_samples,
            LANG_CODE, total_duration, tracker, update
        )
        print(f"  Multi-speaker synthesis with {num_speakers} voices")
    elif speaker_samples:
        first_speaker = next(iter(speaker_samples))
        voice_sample = speaker_samples[first_speaker]
        speech_audio = synthesize_segments(
            translated_segments, voice_sample, LANG_CODE, total_duration, tracker, update
        )
        print(f"  Single-speaker synthesis")
    else:
        speech_audio = synthesize_segments(
            translated_segments, audio_path, LANG_CODE, total_duration, tracker, update
        )
        print(f"  Fallback to original audio")
    print(f"  Speech: {speech_audio}")

    # Stage 7: Mix audio
    print("[7/8] Mixing speech with background...")
    mixed_audio = mix_audio_with_background(speech_audio, background_path, background_volume=0.3)
    print(f"  Mixed: {mixed_audio}")

    # Stage 8: Merge and embed subtitles
    print("[8/8] Generating subtitles and merging video...")
    subtitle_path = generate_srt(translated_segments)
    print(f"  Subtitles: {subtitle_path}")
    safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:50]
    subtitle_lang_code = get_iso_639_2_code(LANG_CODE)
    output = merge_audio_video(
        video_path, mixed_audio, safe_title, tracker, update,
        subtitle_path=subtitle_path, subtitle_lang=subtitle_lang_code
    )
    print(f"  Output: {output}")

    print()
    print("=" * 60)
    print("SUCCESS! Translation complete.")
    print(f"Output file: {output}")
    print("=" * 60)

    # Summary for verification
    print()
    print("VERIFICATION SUMMARY:")
    print(f"  - Multiple speakers: {num_speakers} (need 2+ for multi-speaker test)")
    print(f"  - Voice samples extracted: {list(speaker_samples.keys())}")
    print(f"  - Background preserved: {background_path.exists()}")
    print(f"  - Subtitles generated: {subtitle_path.exists()}")
    print(f"  - Final output exists: {output.exists()}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
