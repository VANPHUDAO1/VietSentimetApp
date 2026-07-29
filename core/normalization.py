"""Normalize text consistently for all models."""

import re
import unicodedata


def normalize_text_default(text: str) -> str:
    """Khớp nguyên văn với normalize_text() trong cả 3 notebook train."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


NORMALIZE_FN_BY_MODEL = {
    "bilstm": normalize_text_default,
    "phobert": normalize_text_default,
    "hybrid": normalize_text_default,
}


def normalize_text(text: str, model_key: str) -> str:
    fn = NORMALIZE_FN_BY_MODEL.get(model_key, normalize_text_default)
    return fn(text)
