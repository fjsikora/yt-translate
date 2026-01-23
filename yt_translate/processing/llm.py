"""
LLM-based translation module using Qwen3-32B.

This module provides high-quality context-aware translations using a local
Qwen3-32B model (4-bit quantized for VRAM efficiency). Falls back to
Google Translate if the LLM fails.

The LLM translation processes segments in batches with overlap to maintain
context continuity across the transcript.
"""

import logging
from typing import Callable, Optional

from deep_translator import GoogleTranslator

from yt_translate.config.constants import (
    LLM_MAX_SEGMENTS_PER_BATCH,
    LLM_BATCH_OVERLAP,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)
from yt_translate.config.languages import GOOGLE_LANG_CODES_BY_NAME as GOOGLE_LANG_CODES

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_llm_model = None
_llm_tokenizer = None


def _load_model():
    """Load Qwen3-32B model with 4-bit quantization.

    Returns:
        Tuple of (model, tokenizer) or (None, None) if loading fails.
    """
    global _llm_model, _llm_tokenizer

    if _llm_model is not None and _llm_tokenizer is not None:
        return _llm_model, _llm_tokenizer

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_name = "Qwen/Qwen3-32B"

        # Configure 4-bit quantization for VRAM efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        logger.info(f"Loading {model_name} with 4-bit quantization...")

        _llm_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        _llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info("Qwen3-32B model loaded successfully")
        return _llm_model, _llm_tokenizer

    except Exception as e:
        logger.warning(f"Failed to load Qwen3-32B model: {e}")
        _llm_model = None
        _llm_tokenizer = None
        return None, None


def _build_translation_prompt(
    segments: list[dict],
    target_lang: str,
    context_before: Optional[list[dict]] = None,
) -> str:
    """Build a translation prompt for the LLM.

    Args:
        segments: List of segments to translate (with 'text' field)
        target_lang: Target language name (e.g., "Spanish")
        context_before: Optional previous segments for context (already translated)

    Returns:
        Formatted prompt string for the LLM.
    """
    prompt_parts = [
        f"You are a professional translator. Translate the following transcript segments from the detected source language to {target_lang}.",
        "",
        "Rules:",
        "1. Preserve the segment numbering exactly as given",
        "2. Maintain natural conversational tone",
        "3. Keep proper nouns, technical terms, and brand names unchanged unless they have standard translations",
        "4. Preserve emotional tone and emphasis",
        "5. Output ONLY the translations in the exact same numbered format",
        "",
    ]

    # Add context from previous batch if available
    if context_before:
        prompt_parts.append("For context, here are the previous segments (already translated):")
        for i, seg in enumerate(context_before[-LLM_BATCH_OVERLAP:], 1):
            translated = seg.get("translated_text", seg.get("text", ""))
            prompt_parts.append(f"[CONTEXT-{i}] {translated}")
        prompt_parts.append("")

    prompt_parts.append("Translate these segments:")
    for i, seg in enumerate(segments, 1):
        text = seg.get("text", "")
        prompt_parts.append(f"[{i}] {text}")

    prompt_parts.extend([
        "",
        "Output translations in format:",
        "[1] translated text here",
        "[2] translated text here",
        "...",
    ])

    return "\n".join(prompt_parts)


def _parse_translation_response(response: str, num_segments: int) -> list[str]:
    """Parse the LLM response to extract translations.

    Args:
        response: Raw LLM response text
        num_segments: Expected number of segments

    Returns:
        List of translated texts, or empty list if parsing fails.
    """
    translations = []

    # Parse numbered lines like [1] translated text
    import re
    pattern = r'\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)'
    matches = re.findall(pattern, response, re.DOTALL)

    # Build a dict of segment number -> translation
    translation_map = {}
    for num_str, text in matches:
        try:
            num = int(num_str)
            translation_map[num] = text.strip()
        except ValueError:
            continue

    # Extract in order
    for i in range(1, num_segments + 1):
        if i in translation_map:
            translations.append(translation_map[i])
        else:
            # Missing segment
            translations.append("")

    return translations if len(translations) == num_segments else []


def _translate_batch_with_llm(
    model,
    tokenizer,
    segments: list[dict],
    target_lang: str,
    context_before: Optional[list[dict]] = None,
) -> list[str]:
    """Translate a batch of segments using the LLM.

    Args:
        model: Loaded LLM model
        tokenizer: Loaded tokenizer
        segments: Segments to translate
        target_lang: Target language name
        context_before: Previous translated segments for context

    Returns:
        List of translated texts, or empty list on failure.
    """
    import torch

    prompt = _build_translation_prompt(segments, target_lang, context_before)

    # Format as chat message for Qwen
    messages = [{"role": "user", "content": prompt}]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Disable thinking mode for faster response
    )

    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate with low temperature for consistent translations
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response (skip input tokens)
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return _parse_translation_response(response, len(segments))


def _translate_with_google(
    text: str,
    target_lang: str,
) -> str:
    """Fallback translation using Google Translate.

    Args:
        text: Text to translate
        target_lang: Target language name or code

    Returns:
        Translated text, or original text on failure.
    """
    lang_code = GOOGLE_LANG_CODES.get(target_lang, target_lang.lower()[:2])

    try:
        return GoogleTranslator(source='auto', target=lang_code).translate(text)
    except Exception as e:
        logger.warning(f"Google Translate failed: {e}")
        return text


def translate_segments_llm(
    segments: list[dict],
    target_lang: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Translate transcript segments using Qwen3-32B LLM.

    This function batches segments according to config settings, maintains
    context across batches via overlap, and falls back to Google Translate
    if the LLM fails.

    Args:
        segments: List of transcript segments with 'text', 'start', 'end', 'duration' fields
        target_lang: Target language name (e.g., "Spanish", "French")
        progress_callback: Optional callback(detail_str) for progress updates

    Returns:
        List of translated segments with added 'original_text' and 'translated_text' fields.
    """
    if not segments:
        return []

    total_segments = len(segments)

    # Report initial status
    if progress_callback:
        progress_callback(f"Preparing to translate {total_segments} segments...")

    # Try to load the LLM
    model, tokenizer = _load_model()
    use_llm = model is not None and tokenizer is not None

    if use_llm:
        if progress_callback:
            progress_callback("Qwen3-32B loaded, translating with LLM...")
        logger.info("Using Qwen3-32B for translation")
    else:
        if progress_callback:
            progress_callback("LLM unavailable, falling back to Google Translate...")
        logger.info("Falling back to Google Translate")

    # Prepare result list
    translated_segments = []

    if use_llm:
        # Process in batches with overlap
        batch_start = 0
        batch_num = 0
        total_batches = (total_segments + LLM_MAX_SEGMENTS_PER_BATCH - 1) // LLM_MAX_SEGMENTS_PER_BATCH

        while batch_start < total_segments:
            batch_end = min(batch_start + LLM_MAX_SEGMENTS_PER_BATCH, total_segments)
            batch_segments = segments[batch_start:batch_end]
            batch_num += 1

            if progress_callback:
                progress_callback(f"Translating batch {batch_num}/{total_batches} ({len(batch_segments)} segments)...")

            # Get context from previous translated segments
            context = translated_segments[-LLM_BATCH_OVERLAP:] if translated_segments else None

            try:
                # Attempt LLM translation
                translations = _translate_batch_with_llm(
                    model, tokenizer, batch_segments, target_lang, context
                )

                if translations and len(translations) == len(batch_segments):
                    # Success - add translated segments
                    for seg, translated_text in zip(batch_segments, translations):
                        translated_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "duration": seg["duration"],
                            "original_text": seg["text"],
                            "translated_text": translated_text,
                        })
                else:
                    # Parsing failed - fall back to Google for this batch
                    logger.warning(f"LLM response parsing failed for batch {batch_num}, using Google Translate")
                    for seg in batch_segments:
                        translated_text = _translate_with_google(seg["text"], target_lang)
                        translated_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "duration": seg["duration"],
                            "original_text": seg["text"],
                            "translated_text": translated_text,
                        })

            except Exception as e:
                # LLM failed - fall back to Google for this batch
                logger.warning(f"LLM translation failed for batch {batch_num}: {e}, using Google Translate")
                for seg in batch_segments:
                    translated_text = _translate_with_google(seg["text"], target_lang)
                    translated_segments.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "duration": seg["duration"],
                        "original_text": seg["text"],
                        "translated_text": translated_text,
                    })

            batch_start = batch_end

    else:
        # Full Google Translate fallback
        for i, seg in enumerate(segments):
            if progress_callback and (i % 10 == 0 or i == total_segments - 1):
                progress_callback(f"Translating segment {i+1}/{total_segments}...")

            translated_text = _translate_with_google(seg["text"], target_lang)
            translated_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "original_text": seg["text"],
                "translated_text": translated_text,
            })

    if progress_callback:
        progress_callback(f"Translated {len(translated_segments)} segments")

    return translated_segments
