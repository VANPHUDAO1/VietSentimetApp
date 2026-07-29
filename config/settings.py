"""
Cấu hình tập trung cho toàn bộ app.
Mọi module khác import hằng số từ đây, KHÔNG hard-code rải rác trong code.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

BILSTM_DIR = CHECKPOINTS_DIR / "bilstm"
PHOBERT_DIR = CHECKPOINTS_DIR / "phobert"
HYBRID_DIR = CHECKPOINTS_DIR / "hybrid"

VNCORENLP_DIR = BASE_DIR / "vncorenlp"
TEENCODE_FILE = BASE_DIR / "data" / "teencode.txt"

MODEL_REGISTRY = {
    "BiLSTM": {
        "key": "bilstm",
        "checkpoint_dir": BILSTM_DIR,
        "loader": "models.bilstm_model:load_bilstm",
        "predictor": "models.bilstm_model:predict_bilstm",
    },
    "PhoBERT": {
        "key": "phobert",
        "checkpoint_dir": PHOBERT_DIR,
        "loader": "models.phobert_model:load_phobert",
        "predictor": "models.phobert_model:predict_phobert",
    },
    "Hybrid PhoBERT-BiLSTM": {
        "key": "hybrid",
        "checkpoint_dir": HYBRID_DIR,
        "loader": "models.hybrid_model:load_hybrid",
        "predictor": "models.hybrid_model:predict_hybrid",
    },
}

LABELS = ["negative", "positive"]
LABEL_DISPLAY = {"negative": "Tiêu cực", "positive": "Tích cực"}
LABEL_COLOR = {"negative": "#e74c3c", "positive": "#2ecc71"}

SAMPLE_SENTENCES = [
    "Giảng viên dạy rất nhiệt tình và dễ hiểu.",
    "Môn học này chán quá, chẳng học được gì cả.",
    "Ừ thì cũng được, không có gì đặc biệt.",
    "Thầy dạy hay lắm... haha đùa đấy, ngủ suốt buổi luôn.",  # sarcasm
    "Cô dạy hơi khó hiểu nma vẫn oke á ạ.",  # teencode
]

LATENCY_WARNING_MS = {
    "cpu": 3000,
    "cuda": 500,
}
