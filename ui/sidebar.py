"""Sidebar for selecting model and compute device."""

import streamlit as st

from config.settings import MODEL_REGISTRY
from core.device_manager import (
    get_device_display_info,
    is_cuda_available,
    resolve_device,
)


def render_sidebar():
    st.sidebar.markdown(
        """
        <div style="background:linear-gradient(135deg,#2563eb,#7c3aed);padding:14px 16px;border-radius:16px;margin-bottom:12px;">
            <div style="font-size:1.05rem;font-weight:700;color:white;">⚙️ Cấu hình hệ thống</div>
            <div style="font-size:0.9rem;color:#e0e7ff;margin-top:4px;">Chọn mô hình và thiết bị phù hợp để bắt đầu phân tích.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.container():
        st.markdown("<div style='padding:8px 4px 4px 4px;'>", unsafe_allow_html=True)
        model_name = st.radio(
            "Chọn model", list(MODEL_REGISTRY.keys()), horizontal=False
        )
        st.markdown("</div>", unsafe_allow_html=True)

        gpu_available = is_cuda_available()
        device_options = ["CPU"] + (["GPU"] if gpu_available else [])
        if not gpu_available:
            st.caption("⚠️ Máy không phát hiện GPU (CUDA) khả dụng.")

        device_choice = st.radio("Thiết bị tính toán", device_options)

        if "last_device_choice" not in st.session_state:
            st.session_state.last_device_choice = device_options[0]

        if device_choice != st.session_state.last_device_choice:
            st.session_state.last_device_choice = device_choice
            st.session_state.show_device_info = True
        elif st.session_state.get("show_device_info", False):
            st.session_state.show_device_info = True

        if st.session_state.get("show_device_info", False):
            st.markdown(
                f"""
                <div style="margin-top:8px;padding:10px 12px;border-radius:12px;background:#f8fafc;border:1px solid #dbeafe;">
                    <div style="font-size:0.9rem;font-weight:600;color:#1d4ed8;">ℹ️ Thông tin thiết bị</div>
                    <div style="font-size:0.84rem;color:#334155;margin-top:4px;">{get_device_display_info(device_choice)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        device_resolution = resolve_device(device_choice)
        if device_resolution.fallback_happened:
            st.warning(device_resolution.message)

    return model_name, device_resolution.resolved
