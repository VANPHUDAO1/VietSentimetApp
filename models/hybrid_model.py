"""Hybrid PhoBERT-BiLSTM model implementation."""

import json
import time

import torch
import torch.nn as nn

from config.settings import LABELS
from core.preprocessing_pipeline import preprocess
from models.base import ModelBundle, PredictionResult


class PhoBERT_BiLSTM_Classifier(nn.Module):
    """Hybrid PhoBERT-BiLSTM classifier."""

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout_prob: float = 0.1,
        num_layers_used: int | None = None,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(model_name)
        if num_layers_used is not None:
            config.num_hidden_layers = num_layers_used

        self.phobert = AutoModel.from_pretrained(model_name, config=config)
        for param in self.phobert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(dropout_prob)
        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        x = self.dropout(sequence_output)
        _, (hidden, _) = self.bilstm(x)
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        logits = self.fc(hidden_cat)
        return logits


def load_hybrid(
    checkpoint_dir: str, device: str = "cpu", model_key: str = "hybrid"
) -> ModelBundle:
    """Load hybrid checkpoint, tokenizer, and config."""
    from transformers import AutoTokenizer

    with open(f"{checkpoint_dir}/config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint_dir}/tokenizer")

    model = PhoBERT_BiLSTM_Classifier(
        model_name=cfg.get("phobert_name", "vinai/phobert-base"),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_classes=cfg.get("num_classes", 2),
        dropout_prob=cfg.get("dropout_prob", 0.1),
        num_layers_used=cfg.get("num_layers_used"),
    )
    state_dict = torch.load(f"{checkpoint_dir}/model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return ModelBundle(
        model=model,
        extra={
            "tokenizer": tokenizer,
            "config": cfg,
            "device": device,
            "model_key": model_key,
        },
    )


def predict_hybrid(text: str, bundle: ModelBundle) -> PredictionResult:
    """Preprocess text and run hybrid inference."""
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
        logits = bundle.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    t2 = time.perf_counter()

    pred_idx = int(probs.argmax())
    return PredictionResult(
        label=LABELS[pred_idx],
        confidence=float(probs[pred_idx]),
        probs={LABELS[i]: float(p) for i, p in enumerate(probs)},
        preprocessing_ms=(t1 - t0) * 1000,
        inference_ms=(t2 - t1) * 1000,
    )
