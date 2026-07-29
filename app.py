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
    page_title="VSFC Sentiment Analysis", page_icon="💬", layout="centered"
)
st.title("💬 Vietnamese Sentiment Analysis")
st.caption("BiLSTM · PhoBERT · Hybrid PhoBERT-BiLSTM — UIT-VSFC")

_env_ok, _env_message = check_environment()
if not _env_ok:
    st.warning(f"⚠️ {_env_message}")

model_name, device = render_sidebar()
text = render_input_panel()

if st.button("🔍 Phân tích", type="primary", use_container_width=True):
    if validate_input(text):
        try:
            bundle = get_model(model_name, device)
            predictor = get_predictor(model_name)
            result = predictor(text, bundle)

            render_result(result, device)
            render_model_info(model_name, bundle.model)

        except (CheckpointMissingError, EnvironmentError_VnCoreNLP) as e:
            render_error(str(e))
        except Exception as e:
            render_error(f"Lỗi không xác định khi dự đoán: {e}")
