"""Render prediction result, probabilities, and latency metrics."""

import streamlit as st

from config.settings import LABEL_COLOR, LABEL_DISPLAY, LATENCY_WARNING_MS
from models.base import PredictionResult


def render_result(result: PredictionResult, device: str):
    label_vi = LABEL_DISPLAY.get(result.label, result.label)
    color = LABEL_COLOR.get(result.label, "#333")

    st.markdown(
        f"""
        <div style="margin-bottom:1.2rem;padding:18px 20px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:22px;box-shadow:0 8px 20px rgba(15,23,42,0.04);">
            <div style="font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:8px;">Văn bản sau tiền xử lý thô</div>
            <div style="color:#334155;line-height:1.7;white-space:pre-wrap;word-break:break-word;">{result.processed_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label" style="color:{color};">{label_vi}</div>
            <div class="result-meta">Độ tin cậy: <strong>{result.confidence:.1%}</strong></div>
        </div>
        """,
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
