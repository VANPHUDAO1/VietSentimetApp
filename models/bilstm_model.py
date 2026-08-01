"""BiLSTM model implementation for inference."""

import json
import time

import torch
import torch.nn as nn

from config.settings import LABELS
from core.preprocessing_pipeline import preprocess
from models.base import ModelBundle, PredictionResult


class BiLSTMClassifier(nn.Module):
    """BiLSTM classifier architecture."""

    def __init__(
        self,
        max_words,
        num_classes,
        embedding_dim=100,
        lstm_units=64,
        dropout1=0.6,
        dropout2=0.5,
    ):
        super().__init__()

        # The embedding matrix shape must match checkpoint weights.
        self.embedding = nn.Embedding(max_words + 1, embedding_dim, padding_idx=0)

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
        )

        self.embed_dropout = nn.Dropout(0.3)
        self.dropout1 = nn.Dropout(dropout1)
        self.fc1 = nn.Linear(lstm_units * 2, 64)  # *2 vì bidirectional
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)
        _, (h_n, _) = self.bilstm(emb)
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        out = self.dropout1(h)
        out = self.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.fc2(out)
        return out


def load_bilstm(
    checkpoint_dir: str, device: str = "cpu", model_key: str = "bilstm"
) -> ModelBundle:
    """Load BiLSTM checkpoint and metadata."""
    with open(f"{checkpoint_dir}/config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(f"{checkpoint_dir}/vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)

    model = BiLSTMClassifier(
        max_words=cfg["max_words"],
        num_classes=cfg["num_classes"],
        embedding_dim=cfg["embedding_dim"],
        lstm_units=cfg.get("lstm_units", 64),
        dropout1=cfg.get("dropout1", 0.6),
        dropout2=cfg.get("dropout2", 0.5),
    )
    state_dict = torch.load(f"{checkpoint_dir}/model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return ModelBundle(
        model=model,
        extra={"vocab": vocab, "config": cfg, "device": device, "model_key": model_key},
    )


def _encode(
    segmented_text: str, vocab: dict, max_len: int, device: str
) -> torch.Tensor:
    word_index = vocab["word_index"]
    oov_id = word_index.get(vocab.get("oov_token", "<OOV>"), 1)

    ids = [word_index.get(tok, oov_id) for tok in segmented_text.split()][:max_len]
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long, device=device)


def predict_bilstm(text: str, bundle: ModelBundle) -> PredictionResult:
    """Preprocess text, run BiLSTM inference, and return prediction metrics."""
    t0 = time.perf_counter()
    result = preprocess(text, model_key=bundle.extra["model_key"])
    t1 = time.perf_counter()

    cfg = bundle.extra["config"]
    device = bundle.extra["device"]
    x = _encode(result.segmented_text, bundle.extra["vocab"], cfg["max_len"], device)

    with torch.no_grad():
        logits = bundle.model(x)
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
