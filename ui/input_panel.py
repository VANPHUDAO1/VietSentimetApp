"""Input panel for text entry and sample sentence selection."""

import streamlit as st

from config.settings import SAMPLE_SENTENCES


def _set_sample_text(sample: str):
    """Set the sample text into the input text area session state."""
    st.session_state.input_text_area = sample


def render_input_panel() -> str:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#eff6ff,#eef2ff);padding:14px 16px;border-radius:16px;border:1px solid #dbeafe;margin-bottom:10px;">
            <div style="font-size:1.1rem;font-weight:700;color:#1e3a8a;">📝 Nhập câu cần phân tích</div>
            <div style="font-size:0.9rem;color:#475569;margin-top:4px;">Chọn nhanh một câu mẫu hoặc tự nhập câu tiếng Việt của bạn.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='margin-bottom:8px;'><strong>Ví dụ nhanh:</strong></div>",
        unsafe_allow_html=True,
    )
    st.caption("Chọn một câu mẫu từ danh sách bên dưới để chèn vào ô nhập.")

    sample_options = [sample["text"] for sample in SAMPLE_SENTENCES]
    selected_sample = st.selectbox(
        "Câu mẫu",
        sample_options,
        index=0,
        key="sample_selectbox",
    )

    if selected_sample:
        selected_text = selected_sample
        if st.button("Chèn câu mẫu", use_container_width=True):
            st.session_state.input_text_area = selected_text

    st.markdown(
        "<div style='margin-top:8px;'><strong>Câu tiếng Việt:</strong></div>",
        unsafe_allow_html=True,
    )
    text = st.text_area(
        "",
        placeholder="Ví dụ: Giảng viên dạy rất nhiệt tình.",
        height=140,
        key="input_text_area",
    )
    return text


def validate_input(text: str) -> bool:
    """Return False and show a warning if input is empty or whitespace."""
    if not text or not text.strip():
        st.warning("Vui lòng nhập câu cần phân tích trước khi bấm nút.")
        return False
    return True
