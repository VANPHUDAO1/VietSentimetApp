"""
Entry point — điều phối toàn bộ UI + core logic.
Chạy: streamlit run app.py
"""

import streamlit as st

from core.model_registry import CheckpointMissingError, get_model, get_predictor
from core.segmentation import EnvironmentError_VnCoreNLP, check_environment
from ui.input_panel import render_input_panel, validate_input
from ui.model_info_panel import render_model_info
from ui.result_panel import render_error, render_result
from ui.sidebar import render_sidebar

st.set_page_config(
    page_title="VSFC Sentiment Analysis",
    page_icon="💬",
    layout="wide",
)

CUSTOM_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f7fbff 0%, #eef4ff 45%, #f8f1ff 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
}
[data-testid="stSidebar"] * {
    color: #f8fafc;
}
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.hero-card {
    background: linear-gradient(135deg, rgba(79,140,255,0.95), rgba(124,77,255,0.95));
    border-radius: 24px;
    padding: 1.4rem 1.6rem;
    color: white;
    box-shadow: 0 12px 30px rgba(79,140,255,0.20);
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 1.7rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.hero-subtitle {
    font-size: 1rem;
    opacity: 0.95;
    line-height: 1.5;
}
.section-card {
    background: rgba(255,255,255,0.85);
    border-radius: 20px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(99,102,241,0.14);
    box-shadow: 0 8px 24px rgba(15,23,42,0.05);
    margin-bottom: 1rem;
}
.result-card {
    background: linear-gradient(120deg, rgba(79,140,255,0.10), rgba(124,77,255,0.10));
    border: 1px solid rgba(79,140,255,0.16);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
}
.result-label {
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.result-meta {
    font-size: 0.95rem;
    color: #334155;
}
.stButton > button {
    border-radius: 999px;
    border: none;
    background: linear-gradient(135deg, #4f8cff, #7c4dff);
    color: white;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(79,140,255,0.25);
}
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    border-radius: 14px;
    border: 1px solid #dfe7ff;
    box-shadow: none;
}
div[data-testid="stMetric"] {
    background: white;
    border-radius: 16px;
    border: 1px solid #edf2ff;
    box-shadow: 0 6px 16px rgba(15,23,42,0.04);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">💬 Demo phân tích cảm xúc tiếng Việt</div>
        <div class="hero-subtitle">
            Nhập câu và xem kết quả trong giao diện.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

_env_ok, _env_message = check_environment()
if not _env_ok:
    st.warning(f"⚠️ {_env_message}")

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
model_name, device = render_sidebar()
text = render_input_panel()

if st.button("🔍 Phân tích", type="primary", use_container_width=True):
    if validate_input(text):
        progress_placeholder = st.empty()
        progress_placeholder.markdown(
            """
            <div style="position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(15,23,42,0.45);z-index:9999;display:flex;align-items:center;justify-content:center;">
                <div style="background:white;padding:24px 28px;border-radius:16px;box-shadow:0 12px 30px rgba(0,0,0,0.2);min-width:320px;">
                    <div style="font-size:1.05rem;font-weight:700;color:#1d4ed8;">⏳ Đang xử lý...</div>
                    <div style="margin-top:10px;color:#475569;">1. Kiểm tra đầu vào</div>
                    <div style="margin-top:6px;color:#475569;">2. Tải model</div>
                    <div style="margin-top:6px;color:#475569;">3. Tiền xử lý</div>
                    <div style="margin-top:6px;color:#475569;">4. Suy luận</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            progress_placeholder.markdown(
                """
                <div style="position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(15,23,42,0.45);z-index:9999;display:flex;align-items:center;justify-content:center;">
                    <div style="background:white;padding:24px 28px;border-radius:16px;box-shadow:0 12px 30px rgba(0,0,0,0.2);min-width:320px;">
                        <div style="font-size:1.05rem;font-weight:700;color:#1d4ed8;">⏳ Đang xử lý...</div>
                        <div style="margin-top:10px;color:#475569;">✅ Kiểm tra đầu vào hoàn tất</div>
                        <div style="margin-top:6px;color:#475569;">2. Tải model</div>
                        <div style="margin-top:6px;color:#475569;">3. Tiền xử lý</div>
                        <div style="margin-top:6px;color:#475569;">4. Suy luận</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            bundle = get_model(model_name, device)
            progress_placeholder.markdown(
                """
                <div style="position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(15,23,42,0.45);z-index:9999;display:flex;align-items:center;justify-content:center;">
                    <div style="background:white;padding:24px 28px;border-radius:16px;box-shadow:0 12px 30px rgba(0,0,0,0.2);min-width:320px;">
                        <div style="font-size:1.05rem;font-weight:700;color:#1d4ed8;">⏳ Đang xử lý...</div>
                        <div style="margin-top:10px;color:#475569;">✅ Kiểm tra đầu vào hoàn tất</div>
                        <div style="margin-top:6px;color:#475569;">✅ Tải model</div>
                        <div style="margin-top:6px;color:#475569;">3. Tiền xử lý</div>
                        <div style="margin-top:6px;color:#475569;">4. Suy luận</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            predictor = get_predictor(model_name)
            progress_placeholder.markdown(
                """
                <div style="position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(15,23,42,0.45);z-index:9999;display:flex;align-items:center;justify-content:center;">
                    <div style="background:white;padding:24px 28px;border-radius:16px;box-shadow:0 12px 30px rgba(0,0,0,0.2);min-width:320px;">
                        <div style="font-size:1.05rem;font-weight:700;color:#1d4ed8;">⏳ Đang xử lý...</div>
                        <div style="margin-top:10px;color:#475569;">✅ Kiểm tra đầu vào hoàn tất</div>
                        <div style="margin-top:6px;color:#475569;">✅ Tải model</div>
                        <div style="margin-top:6px;color:#475569;">✅ Tiền xử lý</div>
                        <div style="margin-top:6px;color:#475569;">4. Suy luận</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            result = predictor(text, bundle)
            progress_placeholder.markdown(
                """
                <div style="position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(15,23,42,0.45);z-index:9999;display:flex;align-items:center;justify-content:center;">
                    <div style="background:white;padding:24px 28px;border-radius:16px;box-shadow:0 12px 30px rgba(0,0,0,0.2);min-width:320px;">
                        <div style="font-size:1.05rem;font-weight:700;color:#1d4ed8;">⏳ Đang xử lý...</div>
                        <div style="margin-top:10px;color:#475569;">✅ Kiểm tra đầu vào hoàn tất</div>
                        <div style="margin-top:6px;color:#475569;">✅ Tải model</div>
                        <div style="margin-top:6px;color:#475569;">✅ Tiền xử lý</div>
                        <div style="margin-top:6px;color:#475569;">✅ Suy luận</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_result(result, device)
            render_model_info(model_name, bundle.model)

        except (CheckpointMissingError, EnvironmentError_VnCoreNLP) as e:
            render_error(str(e))
        except Exception as e:
            render_error(f"Lỗi không xác định khi dự đoán: {e}")
        finally:
            progress_placeholder.empty()

st.markdown("</div>", unsafe_allow_html=True)
