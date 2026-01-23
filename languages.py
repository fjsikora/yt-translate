"""Backwards-compatible entry point for language mappings.

This module re-exports all language constants and functions from
yt_translate.config.languages to maintain backwards compatibility.
"""

from yt_translate.config.languages import *  # noqa: F401, F403
