"""Render prediction result, probabilities, and latency metrics."""

import streamlit as st

from config.settings import LABEL_COLOR, LABEL_DISPLAY, LATENCY_WARNING_MS
from models.base import PredictionResult


def render_result(result: PredictionResult, device: str):
    label_vi = LABEL_DISPLAY.get(result.label, result.label)
    color = LABEL_COLOR.get(result.label, "#333")

    st.markdown(
        f"### Kết quả: <span style='color:{color}'>{label_vi}</span>",
        unsafe_allow_html=True,
    )
    st.metric("Độ tin cậy", f"{result.confidence:.1%}")

    st.bar_chart(result.probs)

    total_ms = result.preprocessing_ms + result.inference_ms
    col1, col2, col3 = st.columns(3)
    col1.metric("Tiền xử lý", f"{result.preprocessing_ms:.0f} ms")
    col2.metric("Inference", f"{result.inference_ms:.0f} ms")
    col3.metric("Tổng", f"{total_ms:.0f} ms")

    warning_threshold = LATENCY_WARNING_MS.get(device, 3000)
    if total_ms > warning_threshold:
        st.caption(
            f"⏱️ Thời gian xử lý cao hơn ngưỡng tham khảo cho {device.upper()} "
            f"({warning_threshold} ms) — bình thường với máy yếu hoặc model lớn."
        )


def render_error(message: str):
    """Show a friendly error message in the UI."""
    st.error(f"⚠️ {message}")
