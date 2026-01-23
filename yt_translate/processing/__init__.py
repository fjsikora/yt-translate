"""Audio and video processing modules."""

from yt_translate.processing.audio import AudioSeparator, SpeakerDiarizer
from yt_translate.processing.lipsync import apply_lip_sync, stretch_video

__all__ = ["AudioSeparator", "SpeakerDiarizer", "apply_lip_sync", "stretch_video"]
