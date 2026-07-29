"""Input panel for text entry and sample sentence selection."""

import streamlit as st

from config.settings import SAMPLE_SENTENCES


def _set_sample_text(sample: str):
    """Set the sample text into the input text area session state."""
    st.session_state.input_text_area = sample


def render_input_panel() -> str:
    st.subheader("Nhập câu cần phân tích")

    st.caption("Hoặc chọn nhanh 1 câu mẫu:")
    cols = st.columns(len(SAMPLE_SENTENCES))
    for col, sample in zip(cols, SAMPLE_SENTENCES):
        short_label = sample[:12] + "…"
        col.button(short_label, help=sample, on_click=_set_sample_text, args=(sample,))

    text = st.text_area(
        "Câu tiếng Việt:",
        placeholder="Ví dụ: Giảng viên dạy rất nhiệt tình.",
        height=100,
        key="input_text_area",
    )
    return text


def validate_input(text: str) -> bool:
    """Return False and show a warning if input is empty or whitespace."""
    if not text or not text.strip():
        st.warning("Vui lòng nhập câu cần phân tích trước khi bấm nút.")
        return False
    return True
