"""Backwards-compatible entry point for translate_video.

This module re-exports all public functions from yt_translate.core.translator
to maintain backwards compatibility with existing code and deployments.
"""

from yt_translate.core.translator import *  # noqa: F401, F403
