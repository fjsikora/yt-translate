"""
Shared language mappings for the yt-translate project.

This module centralizes all language-related dictionaries to eliminate duplication
across cloud_translate.py and translate.py.
"""

# Supported languages mapped by 2-letter ISO 639-1 code to language name
# This is the canonical mapping used throughout the application
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "zh": "Chinese",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tr": "Turkish",
}

# Google Translate language codes (may differ from ISO 639-1)
# Maps 2-letter code to Google's expected code
GOOGLE_LANG_CODES = {
    "ar": "ar",
    "zh": "zh-CN",  # Google uses zh-CN for Simplified Chinese
    "da": "da",
    "nl": "nl",
    "en": "en",
    "fi": "fi",
    "fr": "fr",
    "de": "de",
    "el": "el",
    "he": "he",
    "hi": "hi",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "ms": "ms",
    "no": "no",
    "pl": "pl",
    "pt": "pt",
    "ru": "ru",
    "es": "es",
    "sw": "sw",
    "sv": "sv",
    "tr": "tr",
}

# ISO 639-1 (2-letter) to ISO 639-2 (3-letter) language code mapping
# Used for ffmpeg subtitle metadata which requires 3-letter codes
ISO_639_1_TO_639_2 = {
    "ar": "ara",
    "zh": "chi",
    "da": "dan",
    "nl": "dut",
    "en": "eng",
    "fi": "fin",
    "fr": "fre",
    "de": "ger",
    "el": "gre",
    "he": "heb",
    "hi": "hin",
    "it": "ita",
    "ja": "jpn",
    "ko": "kor",
    "ms": "may",
    "no": "nor",
    "pl": "pol",
    "pt": "por",
    "ru": "rus",
    "es": "spa",
    "sw": "swa",
    "sv": "swe",
    "tr": "tur",
}

# Extended language name mapping for LLM prompts
# Includes additional languages beyond the core supported set
LANG_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


# Inverted mappings for translate.py compatibility (language name -> code)
# translate.py uses language names as keys, e.g., SUPPORTED_LANGUAGES_BY_NAME["Spanish"] -> "es"
SUPPORTED_LANGUAGES_BY_NAME = {name: code for code, name in SUPPORTED_LANGUAGES.items()}

# Google codes by language name for translate.py
# e.g., GOOGLE_LANG_CODES_BY_NAME["Chinese"] -> "zh-CN"
GOOGLE_LANG_CODES_BY_NAME = {
    SUPPORTED_LANGUAGES[code]: google_code
    for code, google_code in GOOGLE_LANG_CODES.items()
}


# Helper functions for common lookups


def get_language_name(code: str) -> str:
    """Get language name from 2-letter code."""
    return SUPPORTED_LANGUAGES.get(code, code)


def get_language_code(name: str) -> str | None:
    """Get 2-letter code from language name (case-insensitive)."""
    name_lower = name.lower()
    for code, lang_name in SUPPORTED_LANGUAGES.items():
        if lang_name.lower() == name_lower:
            return code
    return None


def get_google_code(code: str) -> str:
    """Get Google Translate code from 2-letter ISO code."""
    return GOOGLE_LANG_CODES.get(code, code)


def get_iso_639_2_code(code: str) -> str:
    """Get 3-letter ISO 639-2 code from 2-letter ISO 639-1 code."""
    if len(code) == 3:
        return code
    return ISO_639_1_TO_639_2.get(code, "eng")
