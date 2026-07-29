"""Shared model interfaces and result bundle definitions."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PredictionResult:
    label: str  # "negative" | "positive"
    confidence: float  # confidence score for the predicted label
    probs: dict  # probability distribution over labels
    preprocessing_ms: float
    inference_ms: float


@dataclass
class ModelBundle:
    """Package model and extra resources needed for prediction."""

    model: Any
    extra: dict  # vocab / tokenizer / config tuỳ model
