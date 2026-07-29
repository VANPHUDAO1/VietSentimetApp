"""Shared preprocessing pipeline: raw cleaning, normalization, and segmentation."""

from dataclasses import dataclass

from core.text_cleaning import clean_raw_text
from core.normalization import normalize_text
from core.segmentation import segment_text


@dataclass
class PreprocessResult:
    raw_text: str
    cleaned_text: str
    normalized_text: str
    segmented_text: str


def preprocess(text: str, model_key: str) -> PreprocessResult:
    cleaned = clean_raw_text(text)
    normalized = normalize_text(cleaned, model_key=model_key)
    segmented = segment_text(normalized)

    return PreprocessResult(
        raw_text=text,
        cleaned_text=cleaned,
        normalized_text=normalized,
        segmented_text=segmented,
    )
