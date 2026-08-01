"""PhoBERT model implementation for inference."""

import json
import time

import torch

from config.settings import LABELS
from core.preprocessing_pipeline import preprocess
from models.base import ModelBundle, PredictionResult


def load_phobert(
    checkpoint_dir: str, device: str = "cpu", model_key: str = "phobert"
) -> ModelBundle:
    """Load PhoBERT checkpoint, tokenizer, and optional inference config."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    cfg = {}
    cfg_path = f"{checkpoint_dir}/inference_config.json"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        pass

    return ModelBundle(
        model=model,
        extra={
            "tokenizer": tokenizer,
            "config": cfg,
            "device": device,
            "model_key": model_key,
        },
    )


def predict_phobert(text: str, bundle: ModelBundle) -> PredictionResult:
    """Preprocess text and run PhoBERT inference."""
    t0 = time.perf_counter()
    result = preprocess(text, model_key=bundle.extra["model_key"])
    t1 = time.perf_counter()

    tokenizer = bundle.extra["tokenizer"]
    device = bundle.extra["device"]
    max_len = bundle.extra["config"].get("max_len", 128)

    inputs = tokenizer(
        result.segmented_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len,
    ).to(device)

    with torch.no_grad():
        logits = bundle.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    t2 = time.perf_counter()

    pred_idx = int(probs.argmax())
    return PredictionResult(
        label=LABELS[pred_idx],
        confidence=float(probs[pred_idx]),
        probs={LABELS[i]: float(p) for i, p in enumerate(probs)},
        preprocessing_ms=(t1 - t0) * 1000,
        inference_ms=(t2 - t1) * 1000,
        processed_text=result.cleaned_text,
    )
