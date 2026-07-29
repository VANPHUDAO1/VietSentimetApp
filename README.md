# VSFC Sentiment Analysis — Streamlit Demo

Demo local so sánh 3 model **BiLSTM**, **PhoBERT**, **Hybrid PhoBERT-BiLSTM**
cho bài toán Vietnamese Sentiment Analysis (nhị phân: `negative`/`positive`)
trên bộ dữ liệu UIT-VSFC.

## Kiến trúc source

```
vsfc_streamlit_app/
├── app.py                      # Entry point — điều phối toàn bộ UI + logic
├── config/
│   └── settings.py             # MODEL_REGISTRY, LABELS, đường dẫn checkpoint, câu mẫu demo
├── core/
│   ├── text_cleaning.py        # Giai đoạn 1 — pipeline làm sạch văn bản thô
│   │                            # (12 bước, hợp nhất từ 2 notebook YouTube — xem mục
│   │                            # "Pipeline tiền xử lý — Giai đoạn 1" bên dưới)
│   ├── normalization.py        # Chuẩn hoá text (NFC, lowercase, regex)
│   ├── segmentation.py         # Word segmentation bằng VnCoreNLP + kiểm tra Java
│   ├── preprocessing_pipeline.py  # Gộp 3 bước trên thành 1 pipeline duy nhất
│   ├── device_manager.py       # Chọn CPU/GPU, tự fallback nếu không có CUDA
│   └── model_registry.py       # Nạp & cache model theo cặp (model, device)
├── models/
│   ├── base.py                  # Interface chung: ModelBundle, PredictionResult
│   ├── bilstm_model.py          # load_bilstm() / predict_bilstm()
│   ├── phobert_model.py         # load_phobert() / predict_phobert()
│   └── hybrid_model.py          # load_hybrid() / predict_hybrid()
├── ui/
│   ├── sidebar.py                # Chọn model + chọn thiết bị (CPU/GPU)
│   ├── input_panel.py            # Ô nhập câu + nút câu mẫu dựng sẵn
│   ├── result_panel.py           # Hiển thị nhãn, %confidence, bar chart, latency
│   └── model_info_panel.py       # Thông tin ngắn gọn về model đang chọn
├── checkpoints/                  # (KHÔNG commit — xem .gitignore) chứa file model thật
│   ├── bilstm/
│   ├── phobert/
│   └── hybrid/
├── requirements.txt
└── .gitignore
```

## Pipeline tiền xử lý — Giai đoạn 1

`core/text_cleaning.py` (`clean_raw_text()`) làm sạch input thô người dùng
gõ trực tiếp vào app (emoji, teencode, HTML, URL, gõ lặp ký tự...) — đây là
bước **riêng cho app**, không có sẵn trong pipeline train của 3 model (xem
ghi chú ở mục checklist bên dưới). Output của bước này là input cho
**Giai đoạn 2** (`core/normalization.py` → `core/segmentation.py`), vốn
phải giữ nguyên y hệt lúc train và **không được sửa**.

Pipeline Giai đoạn 1 hiện tại là bản **hợp nhất** từ 2 notebook chuẩn bị dữ
liệu YouTube comment gốc — `..._before_labeling.ipynb` (hàm
`preprocess_comment()`, 9 bước) và `..._after_labeling.ipynb` (hàm
`clean_text()`), vì 2 notebook đó có pipeline lệch nhau khá nhiều (khác
regex URL, khác cách xử lý emoji, thiếu teencode/emoticon ở bản
after_labeling...). Nguyên tắc hợp nhất: lấy `before_labeling` làm nền
chính (giữ nguyên thứ tự + hành vi 9 bước gốc), chỉ chèn thêm 2 phần
`before_labeling` không có nhưng `after_labeling` có, đặt đúng vị trí
tương đối mà `after_labeling` đặt chúng.

Thứ tự đầy đủ 12 bước (đã đối chiếu trực tiếp với source của cả 2 notebook):

| # | Bước | Nguồn | Ghi chú |
|---|---|---|---|
| 1 | `remove_url` | after_labeling (regex nâng cấp) | Bắt thêm email + domain trần (vd `abc.edu.vn`), before_labeling chỉ bắt `http(s)://`/`www.` |
| 2 | `remove_html_tags` | after_labeling (chèn mới) | before_labeling không có bước này |
| 3 | `replace_emoji_with_text` | before_labeling | Thay bằng **nghĩa** tiếng Việt, không xóa thẳng như after_labeling |
| 4 | `normalize_unicode` | before_labeling bước 1/9 | NFC |
| 5 | `remove_timestamp` | before_labeling bước 2/9 | mm:ss / hh:mm:ss |
| 6 | `normalize_punctuation` | before_labeling bước 3/9 | Dấu câu lặp → còn 1 |
| 7 | `normalize_repeated_chars` | before_labeling bước 4/9 | Ký tự lặp ≥3 → rút còn 2 |
| 8 | `to_lowercase` | before_labeling bước 5/9 | |
| 9 | `replace_teencode` | before_labeling bước 6/9 | `flags=IGNORECASE` |
| 10 | `remove_special_chars` | before_labeling bước 7/9 | Xóa emoticon + ký tự đặc biệt, **giữ lại `.,!?`** |
| 11 | `normalize_whitespace` | before_labeling bước 8/9 | |
| 12 | `normalize_vietnamese` | before_labeling bước 9/9 | underthesea |

Bước 4–12 giữ **nguyên văn** thứ tự và hành vi gốc của `before_labeling`,
không đảo vị trí, không sửa hành vi nào — kể cả một bug case-sensitivity
đã biết ở bước 10: `to_lowercase` (bước 8) chạy **trước**
`remove_special_chars` (bước 10), và `remove_special_chars` không dùng
`re.IGNORECASE` khi khớp `EMOTICON_DICT`. Emoticon viết hoa (vd `"XD"`,
`"T_T"`, `"TT"`) sau khi bị lowercase thành `"xd"`, `"t_t"`, `"tt"` sẽ
không còn khớp key gốc trong dict → không bị xóa. Đây là bug **có sẵn
trong chính notebook `before_labeling` gốc**, không phải lỗi riêng của
app, nên **chưa được sửa** trong bản hợp nhất này (yêu cầu ban đầu chỉ là
thống nhất đúng trình tự, không đổi hành vi các bước đã có sẵn).

## Nguyên tắc thiết kế

- **1 pipeline tiền xử lý duy nhất** (`core/preprocessing_pipeline.py`) dùng chung
  cho cả 3 model — tránh lặp lại bug OOV do quên bước normalize/segment trước khi
  tokenize (đã từng gặp trong notebook train gốc).
- **Model registry pattern**: thêm/bớt model chỉ cần sửa `MODEL_REGISTRY` trong
  `config/settings.py`, không đụng vào UI hay logic load/cache.
- **Interface chung** (`models/base.py`): cả 3 model implement đúng
  `load_<model>()` / `predict_<model>()`, trả về `PredictionResult` có sẵn
  `preprocessing_ms` và `inference_ms` tách riêng để phục vụ benchmark latency
  CPU/GPU trong báo cáo.
- **Cache theo `(model, device)`**: đổi model hoặc đổi CPU↔GPU đều tự nạp lại
  đúng, không lẫn lộn giữa các lần chọn.

## Yêu cầu môi trường

- **Python ≥ 3.9** (bắt buộc — code dùng cú pháp generic built-in kiểu
  `dict[tuple[str, str], Any]`, không chạy được trên Python 3.8 trở xuống)
- Java (JDK 8+) cho VnCoreNLP

## Cài đặt

```bash
git clone <repo-url>
cd vsfc_streamlit_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Cài thêm Java (JDK 8+) cho VnCoreNLP, và tải `VnCoreNLP-1.2.jar` + thư mục
`models/` đặt vào `vncorenlp/` ở thư mục gốc project (không commit vào Git).

## Chuẩn bị checkpoint

Copy 3 thư mục output từ notebook train vào đúng vị trí:

```
checkpoints/
├── bilstm/    → model.pt, vocab.json, config.json
├── phobert/   → (kết quả save_pretrained() của model + tokenizer) + inference_config.json
└── hybrid/    → model.pt, config.json, tokenizer/ (thư mục tokenizer save_pretrained())
```

## Chạy app

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`.

## Trạng thái implement (checklist)

- [x] Kiến trúc thư mục + interface chung
- [x] `config/settings.py`, toàn bộ `core/`
- [x] `models/base.py`, `bilstm_model.py`, `phobert_model.py`, `hybrid_model.py`
- [x] Toàn bộ `ui/`, `app.py`
- [x] Rà soát & sửa dead code, docstring sai, bug session_state, hard-code trùng lặp
- [x] **Đối chiếu với cả 3 notebook train thật (07/2026):**
  - `BiLSTMClassifier`: sửa lại đúng kiến trúc thật (có thêm tầng
    `fc1(128→64)` + `ReLU` mà skeleton trước đó thiếu)
  - `PhoBERT_BiLSTM_Classifier`: sửa lại đúng vị trí `Dropout` (đặt TRƯỚC
    BiLSTM, không phải sau), PhoBERT freeze cứng không có tham số bật/tắt
  - `normalize_text()`: xác nhận cả 3 model dùng NGUYÊN VĂN 1 hàm giống hệt
  - **Phát hiện quan trọng:** cả 3 notebook KHÔNG có bước "làm sạch văn bản
    thô" (URL/HTML/ký tự lặp...) — đã bỏ bước này khỏi pipeline mặc định để
    tránh lệch phân phối dữ liệu so với lúc train (xem
    `core/preprocessing_pipeline.py`, `core/text_cleaning.py`)
  - Xác nhận field trong `config.json`/`vocab.json`/`inference_config.json`
    khớp 100% với tên code đang đọc (dựa trên checkpoint thật đã nhận)
- [x] **Hợp nhất pipeline Giai đoạn 1 từ 2 notebook YouTube (07/2026):**
  đối chiếu trực tiếp source của `..._before_labeling.ipynb` và
  `..._after_labeling.ipynb` (2 pipeline lệch nhau đáng kể — khác regex
  URL, khác cách xử lý emoji, thiếu teencode/emoticon ở bản
  after_labeling), viết lại `core/text_cleaning.py` thành 1 pipeline 12
  bước duy nhất, đúng thứ tự từng notebook gốc (chi tiết xem mục
  "Pipeline tiền xử lý — Giai đoạn 1" ở trên). Giữ nguyên bug
  case-sensitivity có sẵn trong notebook gốc (emoticon viết hoa không bị
  xóa) — chưa sửa vì phạm vi công việc chỉ là thống nhất trình tự.
- [ ] Test end-to-end thật với checkpoint đã upload (chưa chạy `streamlit run`
      trong môi trường có đủ Java + VnCoreNLP + checkpoint để xác nhận
      `load_state_dict()` không báo lỗi size-mismatch)

## Liên quan

Dự án phục vụ đề tài luận văn: *"So sánh Bi-LSTM, PhoBERT và Hybrid
PhoBERT-LSTM cho Phân tích cảm xúc tiếng Việt"* — UIT-VSFC dataset.
