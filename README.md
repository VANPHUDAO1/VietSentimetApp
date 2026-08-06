# VSFC Sentiment Analysis — Streamlit Demo

Ứng dụng Streamlit cho demo phân tích cảm xúc tiếng Việt dựa trên 3 mô hình:
- **BiLSTM**
- **PhoBERT**
- **Hybrid PhoBERT-BiLSTM**

App dùng pipeline tiền xử lý chung, model registry và UI trực quan để so sánh
kết quả nhãn `negative` / `positive` trên câu tiếng Việt.

## Tổng quan kiến trúc

- `app.py`: entrypoint Streamlit, quản lý layout, xử lý form và hiển thị kết quả.
- `config/settings.py`: cấu hình trung tâm, đăng ký model, nhãn hiển thị, mẫu câu thử.
- `core/`:
  - `text_cleaning.py`: pipeline làm sạch văn bản thô (URL, HTML, emoji, teencode,
    ký tự lặp, dấu câu,...).
  - `normalization.py`: chuẩn hóa text cho model sau khi làm sạch.
  - `segmentation.py`: phân tách từ tiếng Việt bằng VnCoreNLP và kiểm tra Java.
  - `preprocessing_pipeline.py`: gộp quy trình thành `preprocess(text, model_key)`.
  - `device_manager.py`: chọn CPU/GPU, fallback khi CUDA không khả dụng.
  - `model_registry.py`: load/cache model theo cặp `(model, device)`.
- `models/`:
  - `base.py`: định nghĩa `ModelBundle` và `PredictionResult` dùng chung.
  - `bilstm_model.py`: load/predict BiLSTM.
  - `phobert_model.py`: load/predict PhoBERT.
  - `hybrid_model.py`: load/predict hybrid PhoBERT-BiLSTM.
- `ui/`:
  - `sidebar.py`: chọn model và thiết bị.
  - `input_panel.py`: nhập câu và chèn câu mẫu.
  - `result_panel.py`: hiển thị nhãn, xác suất, biểu đồ và độ trễ.
  - `model_info_panel.py`: hiển thị số tham số model.
- `checkpoints/`: chứa checkpoint mô hình (không commit, xem `.gitignore`).
- `vncorenlp/`: chứa VnCoreNLP jar + models (không commit).

## Yêu cầu

- Python >= 3.9
- Java JDK 8+
- `requirements.txt` gồm:
  - `streamlit>=1.36`
  - `torch>=2.2`
  - `transformers>=4.41`
  - `py-vncorenlp>=0.1.4`
  - `emoji>=2.11`
  - `underthesea>=6.8.0`
  - `pandas>=2.0`

## Cài đặt nhanh

```bash
git clone <repo-url>
cd VietSentMultiModelApp
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Chuẩn bị VnCoreNLP

Cài Java JDK 8+ và đặt thư mục VnCoreNLP vào `vncorenlp/` trong project.
Nội dung cần có:

- `vncorenlp/VnCoreNLP-1.2.jar`
- `vncorenlp/models/` chứa model VnCoreNLP

App sẽ kiểm tra Java và `vncorenlp/` trước khi chạy.

## Chuẩn bị checkpoint

Project không commit checkpoint và các file model nặng. Sao chép thư mục checkpoint
và tokenizer vào cấu trúc sau:

```
checkpoints/
├── bilstm/
│   ├── model.pt
│   ├── vocab.json
│   └── config.json
├── phobert/
│   ├── config.json
│   ├── inference_config.json  # tùy chọn
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── bpe.codes
│   ├── added_tokens.json
│   └── model.safetensors / pytorch_model.bin
└── hybrid/
    ├── model.pt
    ├── config.json
    └── tokenizer/
        ├── tokenizer_config.json
        ├── vocab.txt
        ├── bpe.codes
        └── added_tokens.json
```

`.gitignore` đã được cấu hình để bỏ qua các file checkpoint nặng và tokenizer
liên quan trong `checkpoints/`.

## Chạy ứng dụng

```bash
streamlit run app.py
```

Mở `http://localhost:8501` và:

- chọn model (`BiLSTM`, `PhoBERT`, `Hybrid PhoBERT-BiLSTM`)
- chọn thiết bị (`CPU` hoặc `GPU` nếu có)
- nhập câu tiếng Việt hoặc chèn câu mẫu
- nhấn `Phân tích`

## Hệ thống tiền xử lý

Quy trình tiền xử lý dùng chung cho cả 3 model:

1. `clean_raw_text()` — loại bỏ URL/email/domain, HTML, emoji,
   emoticon, teencode, timestamp, ký tự lặp, dấu câu dư, ký tự đặc biệt.
2. `normalize_text()` — chuẩn hóa text phù hợp với model.
3. `segment_text()` — phân tách từ bằng VnCoreNLP.

Kết quả tiền xử lý hiển thị trong app là phần text sau bước sạch thô (`cleaned_text`).

## Mô tả model

### BiLSTM

- Mô hình `BiLSTMClassifier` với embedding, BiLSTM 2 chiều,
  dropout và 2 tầng fully-connected.
- Dùng `config.json` và `vocab.json` để khởi tạo.
- Input được map sang ID theo vocab, padding đến `max_len`.

### PhoBERT

- Dùng `transformers.AutoTokenizer` và
  `AutoModelForSequenceClassification` từ thư mục checkpoint.
- Thực hiện tokenize text đã phân tách và infer bằng mô hình PhoBERT.

### Hybrid PhoBERT-BiLSTM

- Dùng PhoBERT pretrained frozen để tạo embedding sequence.
- Kết quả embedding đầu vào cho một layer BiLSTM bidirectional rồi
  dự đoán nhãn qua linear layer.
- PhoBERT bị freeze hoàn toàn, chỉ phần BiLSTM/FC được train.

## UI chính

- Sidebar chọn model và thiết bị.
- Input panel gồm ô nhập câu + chọn câu mẫu.
- Kết quả hiển thị:
  - nhãn tiếng Việt
  - xác suất
  - biểu đồ xác suất
  - độ trễ tiền xử lý / inference / tổng
  - văn bản sau tiền xử lý

## Lưu ý

- Model được cache theo cặp `(model, device)` để tránh load lại khi đổi model hay
  giữ lại lựa chọn thiết bị.
- Nếu GPU không khả dụng, app sẽ tự fallback về CPU và thông báo.
- File `checkpoints/` và `vncorenlp/` không nên commit vào Git.

## File chính cần biết

- `app.py`
- `config/settings.py`
- `core/model_registry.py`
- `core/preprocessing_pipeline.py`
- `core/text_cleaning.py`
- `core/segmentation.py`
- `models/bilstm_model.py`
- `models/phobert_model.py`
- `models/hybrid_model.py`
- `ui/sidebar.py`
- `ui/input_panel.py`
- `ui/result_panel.py`
- `ui/model_info_panel.py`

---

Nếu cần mở rộng thêm model mới, cập nhật `MODEL_REGISTRY` trong
`config/settings.py` để thêm loader/predictor mới mà không cần sửa UI.
