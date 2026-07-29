"""Model registry and cache for checkpointed models."""

import importlib
from typing import Any

from config.settings import MODEL_REGISTRY

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


class CheckpointMissingError(RuntimeError):
    """Raised when a model checkpoint is missing."""


def _import_from_path(dotted_path: str):
    module_path, func_name = dotted_path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def get_model(display_name: str, device: str):
    """Load or return a cached ModelBundle for the requested model and device."""
    cache_key = (display_name, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    entry = MODEL_REGISTRY.get(display_name)
    if entry is None:
        raise ValueError(f"Model không tồn tại trong registry: {display_name}")

    checkpoint_dir = entry["checkpoint_dir"]
    if not checkpoint_dir.exists():
        raise CheckpointMissingError(
            f"Thiếu thư mục checkpoint cho model '{display_name}': {checkpoint_dir}"
        )

    loader_fn = _import_from_path(entry["loader"])

    try:
        bundle = loader_fn(str(checkpoint_dir), device=device, model_key=entry["key"])
    except FileNotFoundError as e:
        raise CheckpointMissingError(
            f"Thiếu file checkpoint cần thiết cho model '{display_name}': {e}"
        ) from e

    _MODEL_CACHE[cache_key] = bundle
    return bundle


def get_predictor(display_name: str):
    entry = MODEL_REGISTRY.get(display_name)
    if entry is None:
        raise ValueError(f"Model không tồn tại trong registry: {display_name}")
    return _import_from_path(entry["predictor"])
