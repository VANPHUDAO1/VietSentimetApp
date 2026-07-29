"""Sidebar for selecting model and compute device."""

import streamlit as st

from config.settings import MODEL_REGISTRY
from core.device_manager import is_cuda_available, resolve_device


def render_sidebar():
    st.sidebar.header("Cấu hình")

    model_name = st.sidebar.radio("Chọn model:", list(MODEL_REGISTRY.keys()))

    gpu_available = is_cuda_available()
    device_options = ["CPU"] + (["GPU"] if gpu_available else [])
    if not gpu_available:
        st.sidebar.caption("⚠️ Máy không phát hiện GPU (CUDA) khả dụng.")

    device_choice = st.sidebar.radio("Thiết bị tính toán:", device_options)

    device_resolution = resolve_device(device_choice)
    if device_resolution.fallback_happened:
        st.sidebar.warning(device_resolution.message)

    return model_name, device_resolution.resolved
