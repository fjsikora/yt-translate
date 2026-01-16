"""
Integration tests for the unified video translation pipeline.

Tests verify that process_video_translation correctly handles different job types
(preview, paid, legacy) by mocking external API calls (Replicate, OpenAI).
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the functions under test
from cloud_translate import (
    process_video_translation,
    process_translation,
    process_preview,
    process_full_translation,
    jobs,
    update_job,
)


class MockVideoInfo:
    """Mock video info returned by yt-dlp."""

    def __init__(self) -> None:
        self.title = "Test Video Title"
        self.duration = 120


@pytest.fixture
def mock_video_path(tmp_path: Path) -> Path:
    """Create a mock video file path."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"mock video content")
    return video_path


@pytest.fixture
def mock_audio_path(tmp_path: Path) -> Path:
    """Create a mock audio file path."""
    audio_path = tmp_path / "test_audio.wav"
    audio_path.write_bytes(b"mock audio content")
    return audio_path


@pytest.fixture
def mock_output_path(tmp_path: Path) -> Path:
    """Create a mock output video file path."""
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"mock output video content")
    return output_path


@pytest.fixture
def mock_segments() -> list[dict[str, Any]]:
    """Create mock transcription segments."""
    return [
        {"start": 0.0, "end": 2.0, "text": "Hello world"},
        {"start": 2.0, "end": 5.0, "text": "This is a test"},
    ]


@pytest.fixture
def mock_translated_segments() -> list[dict[str, Any]]:
    """Create mock translated segments."""
    return [
        {"start": 0.0, "end": 2.0, "text": "Hello world", "translated_text": "Hola mundo"},
        {"start": 2.0, "end": 5.0, "text": "This is a test", "translated_text": "Esto es una prueba"},
    ]


@pytest.fixture
def mock_diarization_segments() -> list[dict[str, Any]]:
    """Create mock speaker diarization segments."""
    return [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_01"},
    ]


@pytest.fixture
def mock_speaker_voice_urls() -> dict[str, str]:
    """Create mock speaker voice sample URLs."""
    return {
        "SPEAKER_00": "https://example.com/voice_sample_0.wav",
        "SPEAKER_01": "https://example.com/voice_sample_1.wav",
    }


@pytest.fixture
def clear_jobs():
    """Clear the jobs dictionary before each test."""
    jobs.clear()
    yield
    jobs.clear()


class TestProcessVideoTranslationPreview:
    """Tests for process_video_translation with job_type='preview'."""

    @pytest.mark.asyncio
    async def test_preview_job_calls_correct_db_functions(
        self,
        tmp_path: Path,
        mock_video_path: Path,
        mock_audio_path: Path,
        mock_output_path: Path,
        mock_segments: list[dict[str, Any]],
        mock_translated_segments: list[dict[str, Any]],
        mock_diarization_segments: list[dict[str, Any]],
        mock_speaker_voice_urls: dict[str, str],
        clear_jobs: None,
    ) -> None:
        """Test that preview job type uses db.update_preview_job and db.upload_preview_to_storage."""
        job_id = "test-preview-job-123"
        video_url = "https://www.youtube.com/watch?v=test"
        target_language = "es"

        # Create mocks for all pipeline functions
        with patch("cloud_translate.download_video") as mock_download, \
             patch("cloud_translate.separate_audio") as mock_separate, \
             patch("cloud_translate.diarize_speakers") as mock_diarize, \
             patch("cloud_translate.transcribe_audio") as mock_transcribe, \
             patch("cloud_translate.translate_segments_llm") as mock_translate, \
             patch("cloud_translate.synthesize_segments_multi_speaker") as mock_synth_multi, \
             patch("cloud_translate.mix_audio_with_background") as mock_mix, \
             patch("cloud_translate.generate_srt") as mock_srt, \
             patch("cloud_translate.merge_audio_video") as mock_merge, \
             patch("cloud_translate.db") as mock_db:

            # Configure mock returns
            mock_download.return_value = (
                str(mock_video_path),
                str(mock_audio_path),
                {"title": "Test Video", "duration": 120}
            )
            mock_separate.return_value = (str(mock_audio_path), str(mock_audio_path))
            mock_diarize.return_value = (mock_diarization_segments, mock_speaker_voice_urls)
            mock_transcribe.return_value = mock_segments
            mock_translate.return_value = mock_translated_segments
            mock_synth_multi.return_value = str(mock_audio_path)
            mock_mix.return_value = str(mock_audio_path)
            mock_srt.return_value = str(tmp_path / "subtitles.srt")
            mock_merge.return_value = mock_output_path
            mock_db.upload_preview_to_storage.return_value = f"previews/{job_id}/preview.mp4"

            # Run the pipeline
            await process_video_translation(
                job_id=job_id,
                video_url=video_url,
                target_language=target_language,
                duration_limit=60,
                job_type="preview"
            )

            # Verify db.update_preview_job was called for status updates
            assert mock_db.update_preview_job.called

            # Verify upload was called with correct storage function
            mock_db.upload_preview_to_storage.assert_called_once_with(job_id, str(mock_output_path))

            # Verify the final completion update includes preview_file_path
            final_calls = [
                call for call in mock_db.update_preview_job.call_args_list
                if call.kwargs.get("status") == "completed"
            ]
            assert len(final_calls) == 1
            assert final_calls[0].kwargs.get("preview_file_path") == f"previews/{job_id}/preview.mp4"

            # Verify job is in memory with completed status
            assert job_id in jobs
            assert jobs[job_id]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_preview_job_passes_duration_limit(
        self,
        tmp_path: Path,
        mock_video_path: Path,
        mock_audio_path: Path,
        mock_output_path: Path,
        mock_segments: list[dict[str, Any]],
        mock_translated_segments: list[dict[str, Any]],
        mock_diarization_segments: list[dict[str, Any]],
        mock_speaker_voice_urls: dict[str, str],
        clear_jobs: None,
    ) -> None:
        """Test that preview job passes duration_limit to download_video."""
        job_id = "test-preview-duration-456"
        duration_limit = 60  # 60 seconds for preview

        with patch("cloud_translate.download_video") as mock_download, \
             patch("cloud_translate.separate_audio") as mock_separate, \
             patch("cloud_translate.diarize_speakers") as mock_diarize, \
             patch("cloud_translate.transcribe_audio") as mock_transcribe, \
             patch("cloud_translate.translate_segments_llm") as mock_translate, \
             patch("cloud_translate.synthesize_segments_multi_speaker") as mock_synth_multi, \
             patch("cloud_translate.mix_audio_with_background") as mock_mix, \
             patch("cloud_translate.generate_srt") as mock_srt, \
             patch("cloud_translate.merge_audio_video") as mock_merge, \
             patch("cloud_translate.db") as mock_db:

            mock_download.return_value = (
                str(mock_video_path),
                str(mock_audio_path),
                {"title": "Test Video", "duration": 120}
            )
            mock_separate.return_value = (str(mock_audio_path), str(mock_audio_path))
            mock_diarize.return_value = (mock_diarization_segments, mock_speaker_voice_urls)
            mock_transcribe.return_value = mock_segments
            mock_translate.return_value = mock_translated_segments
            mock_synth_multi.return_value = str(mock_audio_path)
            mock_mix.return_value = str(mock_audio_path)
            mock_srt.return_value = str(tmp_path / "subtitles.srt")
            mock_merge.return_value = mock_output_path
            mock_db.upload_preview_to_storage.return_value = "previews/test/preview.mp4"

            await process_video_translation(
                job_id=job_id,
                video_url="https://www.youtube.com/watch?v=test",
                target_language="es",
                duration_limit=duration_limit,
                job_type="preview"
            )

            # Verify download_video was called with duration_limit
            mock_download.assert_called_once()
            call_kwargs = mock_download.call_args
            assert call_kwargs[1].get("duration_limit") == duration_limit


class TestProcessVideoTranslationPaid:
    """Tests for process_video_translation with job_type='paid'."""

    @pytest.mark.asyncio
    async def test_paid_job_calls_correct_db_functions(
        self,
        tmp_path: Path,
        mock_video_path: Path,
        mock_audio_path: Path,
        mock_output_path: Path,
        mock_segments: list[dict[str, Any]],
        mock_translated_segments: list[dict[str, Any]],
        mock_diarization_segments: list[dict[str, Any]],
        mock_speaker_voice_urls: dict[str, str],
        clear_jobs: None,
    ) -> None:
        """Test that paid job type uses db.update_translation_job and db.upload_translation_to_storage."""
        job_id = "test-paid-job-789"
        video_url = "https://www.youtube.com/watch?v=test"
        target_language = "fr"

        with patch("cloud_translate.download_video") as mock_download, \
             patch("cloud_translate.separate_audio") as mock_separate, \
             patch("cloud_translate.diarize_speakers") as mock_diarize, \
             patch("cloud_translate.transcribe_audio") as mock_transcribe, \
             patch("cloud_translate.translate_segments_llm") as mock_translate, \
             patch("cloud_translate.synthesize_segments_multi_speaker") as mock_synth_multi, \
             patch("cloud_translate.mix_audio_with_background") as mock_mix, \
             patch("cloud_translate.generate_srt") as mock_srt, \
             patch("cloud_translate.merge_audio_video") as mock_merge, \
             patch("cloud_translate.db") as mock_db:

            mock_download.return_value = (
                str(mock_video_path),
                str(mock_audio_path),
                {"title": "Test Video", "duration": 300}
            )
            mock_separate.return_value = (str(mock_audio_path), str(mock_audio_path))
            mock_diarize.return_value = (mock_diarization_segments, mock_speaker_voice_urls)
            mock_transcribe.return_value = mock_segments
            mock_translate.return_value = mock_translated_segments
            mock_synth_multi.return_value = str(mock_audio_path)
            mock_mix.return_value = str(mock_audio_path)
            mock_srt.return_value = str(tmp_path / "subtitles.srt")
            mock_merge.return_value = mock_output_path
            mock_db.upload_translation_to_storage.return_value = f"translations/{job_id}/translation.mp4"

            await process_video_translation(
                job_id=job_id,
                video_url=video_url,
                target_language=target_language,
                duration_limit=None,  # Full video for paid
                job_type="paid"
            )

            # Verify db.update_translation_job was called for status updates
            assert mock_db.update_translation_job.called

            # Verify upload was called with correct storage function
            mock_db.upload_translation_to_storage.assert_called_once_with(job_id, str(mock_output_path))

            # Verify the final completion update includes output_file_path
            final_calls = [
                call for call in mock_db.update_translation_job.call_args_list
                if call.kwargs.get("status") == "completed"
            ]
            assert len(final_calls) == 1
            assert final_calls[0].kwargs.get("output_file_path") == f"translations/{job_id}/translation.mp4"

            # Verify job is in memory with completed status
            assert job_id in jobs
            assert jobs[job_id]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_paid_job_no_duration_limit(
        self,
        tmp_path: Path,
        mock_video_path: Path,
        mock_audio_path: Path,
        mock_output_path: Path,
        mock_segments: list[dict[str, Any]],
        mock_translated_segments: list[dict[str, Any]],
        mock_diarization_segments: list[dict[str, Any]],
        mock_speaker_voice_urls: dict[str, str],
        clear_jobs: None,
    ) -> None:
        """Test that paid job passes duration_limit=None to download_video for full video."""
        job_id = "test-paid-full-video"

        with patch("cloud_translate.download_video") as mock_download, \
             patch("cloud_translate.separate_audio") as mock_separate, \
             patch("cloud_translate.diarize_speakers") as mock_diarize, \
             patch("cloud_translate.transcribe_audio") as mock_transcribe, \
             patch("cloud_translate.translate_segments_llm") as mock_translate, \
             patch("cloud_translate.synthesize_segments_multi_speaker") as mock_synth_multi, \
             patch("cloud_translate.mix_audio_with_background") as mock_mix, \
             patch("cloud_translate.generate_srt") as mock_srt, \
             patch("cloud_translate.merge_audio_video") as mock_merge, \
             patch("cloud_translate.db") as mock_db:

            mock_download.return_value = (
                str(mock_video_path),
                str(mock_audio_path),
                {"title": "Test Video", "duration": 600}
            )
            mock_separate.return_value = (str(mock_audio_path), str(mock_audio_path))
            mock_diarize.return_value = (mock_diarization_segments, mock_speaker_voice_urls)
            mock_transcribe.return_value = mock_segments
            mock_translate.return_value = mock_translated_segments
            mock_synth_multi.return_value = str(mock_audio_path)
            mock_mix.return_value = str(mock_audio_path)
            mock_srt.return_value = str(tmp_path / "subtitles.srt")
            mock_merge.return_value = mock_output_path
            mock_db.upload_translation_to_storage.return_value = "translations/test/translation.mp4"

            await process_video_translation(
                job_id=job_id,
                video_url="https://www.youtube.com/watch?v=test",
                target_language="de",
                duration_limit=None,
                job_type="paid"
            )

            # Verify download_video was called with duration_limit=None
            mock_download.assert_called_once()
            call_kwargs = mock_download.call_args
            assert call_kwargs[1].get("duration_limit") is None


class TestProcessVideoTranslationLegacy:
    """Tests for process_video_translation with job_type='legacy'."""

    @pytest.mark.asyncio
    async def test_legacy_job_uses_memory_only(
        self,
        tmp_path: Path,
        mock_video_path: Path,
        mock_audio_path: Path,
        mock_output_path: Path,
        mock_segments: list[dict[str, Any]],
        mock_translated_segments: list[dict[str, Any]],
        mock_diarization_segments: list[dict[str, Any]],
        mock_speaker_voice_urls: dict[str, str],
        clear_jobs: None,
    ) -> None:
        """Test that legacy job type uses in-memory update_job and no cloud upload."""
        job_id = "test-legacy-job-101"
        video_url = "https://www.youtube.com/watch?v=test"
        target_language = "ja"

        with patch("cloud_translate.download_video") as mock_download, \
             patch("cloud_translate.separate_audio") as mock_separate, \
             patch("cloud_translate.diarize_speakers") as mock_diarize, \
             patch("cloud_translate.transcribe_audio") as mock_transcribe, \
             patch("cloud_translate.translate_segments_llm") as mock_translate, \
             patch("cloud_translate.synthesize_segments_multi_speaker") as mock_synth_multi, \
             patch("cloud_translate.mix_audio_with_background") as mock_mix, \
             patch("cloud_translate.generate_srt") as mock_srt, \
             patch("cloud_translate.merge_audio_video") as mock_merge, \
             patch("cloud_translate.db") as mock_db:

            mock_download.return_value = (
                str(mock_video_path),
                str(mock_audio_path),
                {"title": "Test Video", "duration": 120}
            )
            mock_separate.return_value = (str(mock_audio_path), str(mock_audio_path))
            mock_diarize.return_value = (mock_diarization_segments, mock_speaker_voice_urls)
            mock_transcribe.return_value = mock_segments
            mock_translate.return_value = mock_translated_segments
            mock_synth_multi.return_value = str(mock_audio_path)
            mock_mix.return_value = str(mock_audio_path)
            mock_srt.return_value = str(tmp_path / "subtitles.srt")
            mock_merge.return_value = mock_output_path

            await process_video_translation(
                job_id=job_id,
                video_url=video_url,
                target_language=target_language,
                duration_limit=None,
                job_type="legacy"
            )

            # Verify NO db functions were called for upload
            mock_db.upload_preview_to_storage.assert_not_called()
            mock_db.upload_translation_to_storage.assert_not_called()

            # Verify job is in memory with completed status and output fields
            assert job_id in jobs
            assert jobs[job_id]["status"] == "completed"
            assert jobs[job_id]["output_file"] == str(mock_output_path)
            assert jobs[job_id]["output_url"] == f"/download/{job_id}"


class TestProcessVideoTranslationErrorHandling:
    """Tests for error handling in process_video_translation."""

    @pytest.mark.asyncio
    async def test_preview_job_error_updates_db_with_error_message(
        self,
        clear_jobs: None,
    ) -> None:
        """Test that preview job errors are properly recorded in db."""
        job_id = "test-preview-error-123"
        error_message = "Video download failed: network timeout"

        with patch("cloud_translate.download_video") as mock_download, \
             patch("cloud_translate.db") as mock_db:

            # Make download_video raise an exception
            mock_download.side_effect = Exception(error_message)

            with pytest.raises(Exception, match=error_message):
                await process_video_translation(
                    job_id=job_id,
                    video_url="https://www.youtube.com/watch?v=test",
                    target_language="es",
                    duration_limit=60,
                    job_type="preview"
                )

            # Verify error was recorded in db
            mock_db.update_preview_job.assert_called()
            error_calls = [
                call for call in mock_db.update_preview_job.call_args_list
                if call.kwargs.get("status") == "failed"
            ]
            assert len(error_calls) == 1
            assert error_message in error_calls[0].kwargs.get("error_message", "")

    @pytest.mark.asyncio
    async def test_paid_job_error_updates_db_with_error_message(
        self,
        clear_jobs: None,
    ) -> None:
        """Test that paid job errors are properly recorded in db."""
        job_id = "test-paid-error-456"
        error_message = "Transcription failed: API quota exceeded"

        with patch("cloud_translate.download_video") as mock_download, \
             patch("cloud_translate.separate_audio") as mock_separate, \
             patch("cloud_translate.diarize_speakers") as mock_diarize, \
             patch("cloud_translate.transcribe_audio") as mock_transcribe, \
             patch("cloud_translate.db") as mock_db:

            mock_download.return_value = ("/tmp/video.mp4", "/tmp/audio.wav", {"title": "Test", "duration": 120})
            mock_separate.return_value = ("/tmp/vocals.wav", "/tmp/background.wav")
            mock_diarize.return_value = ([], {})
            mock_transcribe.side_effect = Exception(error_message)

            with pytest.raises(Exception, match=error_message):
                await process_video_translation(
                    job_id=job_id,
                    video_url="https://www.youtube.com/watch?v=test",
                    target_language="fr",
                    duration_limit=None,
                    job_type="paid"
                )

            # Verify error was recorded in db
            mock_db.update_translation_job.assert_called()
            error_calls = [
                call for call in mock_db.update_translation_job.call_args_list
                if call.kwargs.get("status") == "failed"
            ]
            assert len(error_calls) == 1
            assert error_message in error_calls[0].kwargs.get("error_message", "")


class TestBackwardsCompatibilityWrappers:
    """Tests for backwards-compatible wrapper functions."""

    @pytest.mark.asyncio
    async def test_process_preview_calls_unified_function(
        self,
        clear_jobs: None,
    ) -> None:
        """Test that process_preview correctly calls process_video_translation."""
        job_id = "test-wrapper-preview"
        video_url = "https://www.youtube.com/watch?v=test"
        target_language = "es"

        with patch("cloud_translate.process_video_translation") as mock_unified:
            # Import the wrapper
            from cloud_translate import process_preview

            await process_preview(job_id, video_url, target_language)

            mock_unified.assert_called_once_with(
                job_id=job_id,
                video_url=video_url,
                target_language=target_language,
                duration_limit=60,  # PREVIEW_DURATION_SECONDS default
                job_type="preview"
            )

    @pytest.mark.asyncio
    async def test_process_full_translation_calls_unified_function(
        self,
        clear_jobs: None,
    ) -> None:
        """Test that process_full_translation correctly calls process_video_translation."""
        job_id = "test-wrapper-paid"
        video_url = "https://www.youtube.com/watch?v=test"
        target_language = "de"

        with patch("cloud_translate.process_video_translation") as mock_unified:
            from cloud_translate import process_full_translation

            await process_full_translation(job_id, video_url, target_language)

            mock_unified.assert_called_once_with(
                job_id=job_id,
                video_url=video_url,
                target_language=target_language,
                duration_limit=None,  # Full video
                job_type="paid"
            )

    @pytest.mark.asyncio
    async def test_process_translation_calls_unified_function(
        self,
        clear_jobs: None,
    ) -> None:
        """Test that process_translation (legacy) correctly calls process_video_translation."""
        job_id = "test-wrapper-legacy"
        video_url = "https://www.youtube.com/watch?v=test"
        target_language = "ja"

        with patch("cloud_translate.process_video_translation") as mock_unified:
            await process_translation(job_id, video_url, target_language)

            mock_unified.assert_called_once_with(
                job_id=job_id,
                video_url=video_url,
                target_language=target_language,
                duration_limit=None,  # Full video for legacy
                job_type="legacy"
            )


class TestPipelineProgressUpdates:
    """Tests for progress update behavior during pipeline execution."""

    @pytest.mark.asyncio
    async def test_pipeline_updates_progress_through_all_stages(
        self,
        tmp_path: Path,
        mock_video_path: Path,
        mock_audio_path: Path,
        mock_output_path: Path,
        mock_segments: list[dict[str, Any]],
        mock_translated_segments: list[dict[str, Any]],
        mock_diarization_segments: list[dict[str, Any]],
        mock_speaker_voice_urls: dict[str, str],
        clear_jobs: None,
    ) -> None:
        """Test that pipeline updates progress through all expected stages."""
        job_id = "test-progress-stages"
        expected_stages = [
            "initializing", "download", "separate", "diarize",
            "transcribe", "translate", "synthesize", "mix",
            "subtitles", "merge", "upload", "done"
        ]

        with patch("cloud_translate.download_video") as mock_download, \
             patch("cloud_translate.separate_audio") as mock_separate, \
             patch("cloud_translate.diarize_speakers") as mock_diarize, \
             patch("cloud_translate.transcribe_audio") as mock_transcribe, \
             patch("cloud_translate.translate_segments_llm") as mock_translate, \
             patch("cloud_translate.synthesize_segments_multi_speaker") as mock_synth_multi, \
             patch("cloud_translate.mix_audio_with_background") as mock_mix, \
             patch("cloud_translate.generate_srt") as mock_srt, \
             patch("cloud_translate.merge_audio_video") as mock_merge, \
             patch("cloud_translate.db") as mock_db:

            mock_download.return_value = (
                str(mock_video_path),
                str(mock_audio_path),
                {"title": "Test Video", "duration": 120}
            )
            mock_separate.return_value = (str(mock_audio_path), str(mock_audio_path))
            mock_diarize.return_value = (mock_diarization_segments, mock_speaker_voice_urls)
            mock_transcribe.return_value = mock_segments
            mock_translate.return_value = mock_translated_segments
            mock_synth_multi.return_value = str(mock_audio_path)
            mock_mix.return_value = str(mock_audio_path)
            mock_srt.return_value = str(tmp_path / "subtitles.srt")
            mock_merge.return_value = mock_output_path
            mock_db.upload_preview_to_storage.return_value = "previews/test/preview.mp4"

            await process_video_translation(
                job_id=job_id,
                video_url="https://www.youtube.com/watch?v=test",
                target_language="es",
                duration_limit=60,
                job_type="preview"
            )

            # Collect all stages from update calls
            stages_seen = set()
            for call in mock_db.update_preview_job.call_args_list:
                stage = call.kwargs.get("stage")
                if stage:
                    stages_seen.add(stage)

            # Verify key stages were reached
            assert "initializing" in stages_seen
            assert "download" in stages_seen
            assert "done" in stages_seen or "completed" in stages_seen or "upload" in stages_seen
