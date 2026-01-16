"""
Unit tests for the languages module.

Verifies that language mappings are complete, consistent, and use valid ISO codes.
"""

import pytest

from languages import (
    GOOGLE_LANG_CODES,
    ISO_639_1_TO_639_2,
    LANG_NAMES,
    SUPPORTED_LANGUAGES,
    SUPPORTED_LANGUAGES_BY_NAME,
    GOOGLE_LANG_CODES_BY_NAME,
    get_language_name,
    get_language_code,
    get_google_code,
    get_iso_639_2_code,
)


# Valid ISO 639-1 codes (subset of the full standard, covering all codes used in project)
# Source: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
VALID_ISO_639_1_CODES = {
    "ar",  # Arabic
    "bg",  # Bulgarian
    "cs",  # Czech
    "da",  # Danish
    "de",  # German
    "el",  # Greek
    "en",  # English
    "es",  # Spanish
    "et",  # Estonian
    "fi",  # Finnish
    "fr",  # French
    "he",  # Hebrew
    "hi",  # Hindi
    "hr",  # Croatian
    "hu",  # Hungarian
    "id",  # Indonesian
    "it",  # Italian
    "ja",  # Japanese
    "ko",  # Korean
    "lt",  # Lithuanian
    "lv",  # Latvian
    "ms",  # Malay
    "nl",  # Dutch
    "no",  # Norwegian
    "pl",  # Polish
    "pt",  # Portuguese
    "ro",  # Romanian
    "ru",  # Russian
    "sk",  # Slovak
    "sl",  # Slovenian
    "sr",  # Serbian
    "sv",  # Swedish
    "sw",  # Swahili
    "th",  # Thai
    "tr",  # Turkish
    "uk",  # Ukrainian
    "vi",  # Vietnamese
    "zh",  # Chinese
}


class TestSupportedLanguagesConsistency:
    """Tests verifying consistency between SUPPORTED_LANGUAGES and other mappings."""

    def test_all_supported_languages_have_google_codes(self) -> None:
        """Every key in SUPPORTED_LANGUAGES must exist in GOOGLE_LANG_CODES."""
        missing_codes = []
        for lang_code in SUPPORTED_LANGUAGES:
            if lang_code not in GOOGLE_LANG_CODES:
                missing_codes.append(lang_code)

        assert not missing_codes, (
            f"The following language codes from SUPPORTED_LANGUAGES are missing in "
            f"GOOGLE_LANG_CODES: {missing_codes}"
        )

    def test_all_supported_languages_have_iso_639_2_codes(self) -> None:
        """Every key in SUPPORTED_LANGUAGES must exist in ISO_639_1_TO_639_2."""
        missing_codes = []
        for lang_code in SUPPORTED_LANGUAGES:
            if lang_code not in ISO_639_1_TO_639_2:
                missing_codes.append(lang_code)

        assert not missing_codes, (
            f"The following language codes from SUPPORTED_LANGUAGES are missing in "
            f"ISO_639_1_TO_639_2: {missing_codes}"
        )

    def test_all_supported_languages_have_lang_names(self) -> None:
        """Every key in SUPPORTED_LANGUAGES must exist in LANG_NAMES."""
        missing_codes = []
        for lang_code in SUPPORTED_LANGUAGES:
            if lang_code not in LANG_NAMES:
                missing_codes.append(lang_code)

        assert not missing_codes, (
            f"The following language codes from SUPPORTED_LANGUAGES are missing in "
            f"LANG_NAMES: {missing_codes}"
        )

    def test_lang_names_consistent_with_supported_languages(self) -> None:
        """LANG_NAMES values should match SUPPORTED_LANGUAGES values for common keys."""
        inconsistencies = []
        for lang_code in SUPPORTED_LANGUAGES:
            if lang_code in LANG_NAMES:
                if SUPPORTED_LANGUAGES[lang_code] != LANG_NAMES[lang_code]:
                    inconsistencies.append(
                        f"{lang_code}: SUPPORTED_LANGUAGES='{SUPPORTED_LANGUAGES[lang_code]}' "
                        f"vs LANG_NAMES='{LANG_NAMES[lang_code]}'"
                    )

        assert not inconsistencies, (
            f"Language name inconsistencies found:\n" + "\n".join(inconsistencies)
        )


class TestIso639Codes:
    """Tests verifying that language codes follow ISO 639 standards."""

    def test_supported_languages_use_valid_iso_639_1_codes(self) -> None:
        """All keys in SUPPORTED_LANGUAGES must be valid ISO 639-1 codes."""
        invalid_codes = []
        for lang_code in SUPPORTED_LANGUAGES:
            if lang_code not in VALID_ISO_639_1_CODES:
                invalid_codes.append(lang_code)

        assert not invalid_codes, (
            f"The following codes in SUPPORTED_LANGUAGES are not valid ISO 639-1 codes: {invalid_codes}"
        )

    def test_google_lang_codes_use_valid_iso_639_1_keys(self) -> None:
        """All keys in GOOGLE_LANG_CODES must be valid ISO 639-1 codes."""
        invalid_codes = []
        for lang_code in GOOGLE_LANG_CODES:
            if lang_code not in VALID_ISO_639_1_CODES:
                invalid_codes.append(lang_code)

        assert not invalid_codes, (
            f"The following keys in GOOGLE_LANG_CODES are not valid ISO 639-1 codes: {invalid_codes}"
        )

    def test_lang_names_use_valid_iso_639_1_codes(self) -> None:
        """All keys in LANG_NAMES must be valid ISO 639-1 codes."""
        invalid_codes = []
        for lang_code in LANG_NAMES:
            if lang_code not in VALID_ISO_639_1_CODES:
                invalid_codes.append(lang_code)

        assert not invalid_codes, (
            f"The following keys in LANG_NAMES are not valid ISO 639-1 codes: {invalid_codes}"
        )

    def test_iso_639_1_to_639_2_keys_are_valid(self) -> None:
        """All keys in ISO_639_1_TO_639_2 must be valid ISO 639-1 codes."""
        invalid_codes = []
        for lang_code in ISO_639_1_TO_639_2:
            if lang_code not in VALID_ISO_639_1_CODES:
                invalid_codes.append(lang_code)

        assert not invalid_codes, (
            f"The following keys in ISO_639_1_TO_639_2 are not valid ISO 639-1 codes: {invalid_codes}"
        )

    def test_iso_639_2_values_are_three_letters(self) -> None:
        """All values in ISO_639_1_TO_639_2 must be 3-letter codes."""
        invalid_values = []
        for lang_code, iso_639_2 in ISO_639_1_TO_639_2.items():
            if len(iso_639_2) != 3:
                invalid_values.append(f"{lang_code}: '{iso_639_2}'")

        assert not invalid_values, (
            f"The following ISO 639-2 values are not 3 letters: {invalid_values}"
        )


class TestInvertedMappings:
    """Tests for the inverted (by-name) mappings."""

    def test_supported_languages_by_name_has_all_languages(self) -> None:
        """SUPPORTED_LANGUAGES_BY_NAME should have all language names from SUPPORTED_LANGUAGES."""
        for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
            assert lang_name in SUPPORTED_LANGUAGES_BY_NAME, (
                f"Language '{lang_name}' ({lang_code}) not found in SUPPORTED_LANGUAGES_BY_NAME"
            )
            assert SUPPORTED_LANGUAGES_BY_NAME[lang_name] == lang_code

    def test_google_lang_codes_by_name_has_all_languages(self) -> None:
        """GOOGLE_LANG_CODES_BY_NAME should have all language names from SUPPORTED_LANGUAGES."""
        for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
            assert lang_name in GOOGLE_LANG_CODES_BY_NAME, (
                f"Language '{lang_name}' ({lang_code}) not found in GOOGLE_LANG_CODES_BY_NAME"
            )
            assert GOOGLE_LANG_CODES_BY_NAME[lang_name] == GOOGLE_LANG_CODES[lang_code]


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_language_name_known_code(self) -> None:
        """get_language_name returns correct name for known codes."""
        assert get_language_name("en") == "English"
        assert get_language_name("es") == "Spanish"
        assert get_language_name("zh") == "Chinese"
        assert get_language_name("ja") == "Japanese"

    def test_get_language_name_unknown_code(self) -> None:
        """get_language_name returns the code itself for unknown codes."""
        assert get_language_name("xx") == "xx"
        assert get_language_name("unknown") == "unknown"

    def test_get_language_code_known_name(self) -> None:
        """get_language_code returns correct code for known names."""
        assert get_language_code("English") == "en"
        assert get_language_code("Spanish") == "es"
        assert get_language_code("Chinese") == "zh"

    def test_get_language_code_case_insensitive(self) -> None:
        """get_language_code is case-insensitive."""
        assert get_language_code("english") == "en"
        assert get_language_code("SPANISH") == "es"
        assert get_language_code("ChInEsE") == "zh"

    def test_get_language_code_unknown_name(self) -> None:
        """get_language_code returns None for unknown names."""
        assert get_language_code("Unknown") is None
        assert get_language_code("Klingon") is None

    def test_get_google_code_known_code(self) -> None:
        """get_google_code returns correct Google code for known codes."""
        assert get_google_code("en") == "en"
        assert get_google_code("es") == "es"
        assert get_google_code("zh") == "zh-CN"  # Chinese has special mapping

    def test_get_google_code_unknown_code(self) -> None:
        """get_google_code returns the code itself for unknown codes."""
        assert get_google_code("xx") == "xx"

    def test_get_iso_639_2_code_known_code(self) -> None:
        """get_iso_639_2_code returns correct 3-letter code for known codes."""
        assert get_iso_639_2_code("en") == "eng"
        assert get_iso_639_2_code("es") == "spa"
        assert get_iso_639_2_code("zh") == "chi"
        assert get_iso_639_2_code("de") == "ger"
        assert get_iso_639_2_code("fr") == "fre"

    def test_get_iso_639_2_code_already_three_letter(self) -> None:
        """get_iso_639_2_code returns the code if already 3 letters."""
        assert get_iso_639_2_code("eng") == "eng"
        assert get_iso_639_2_code("spa") == "spa"

    def test_get_iso_639_2_code_unknown_defaults_to_eng(self) -> None:
        """get_iso_639_2_code returns 'eng' for unknown codes."""
        assert get_iso_639_2_code("xx") == "eng"


class TestDataIntegrity:
    """Tests for data integrity and expected content."""

    def test_supported_languages_count(self) -> None:
        """Verify expected number of supported languages."""
        # Should have 23 languages as defined in languages.py
        assert len(SUPPORTED_LANGUAGES) == 23

    def test_expected_languages_present(self) -> None:
        """Verify key languages are present in SUPPORTED_LANGUAGES."""
        expected_languages = [
            ("en", "English"),
            ("es", "Spanish"),
            ("fr", "French"),
            ("de", "German"),
            ("zh", "Chinese"),
            ("ja", "Japanese"),
            ("ko", "Korean"),
            ("ru", "Russian"),
            ("ar", "Arabic"),
        ]
        for code, name in expected_languages:
            assert code in SUPPORTED_LANGUAGES, f"Language code '{code}' should be in SUPPORTED_LANGUAGES"
            assert SUPPORTED_LANGUAGES[code] == name, (
                f"Expected SUPPORTED_LANGUAGES['{code}'] to be '{name}', "
                f"got '{SUPPORTED_LANGUAGES[code]}'"
            )

    def test_chinese_google_code_is_zh_cn(self) -> None:
        """Chinese should map to zh-CN for Google Translate (Simplified Chinese)."""
        assert GOOGLE_LANG_CODES["zh"] == "zh-CN"
