"""Show model summary and parameter counts."""

import streamlit as st


def render_model_info(display_name: str, model):
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with st.expander(f"ℹ️ Thông tin model: {display_name}"):
        st.write(f"- Tổng số tham số: **{num_params:,}**")
        st.write(f"- Số tham số có thể train: **{trainable_params:,}**")
        if display_name == "Hybrid PhoBERT-BiLSTM":
            st.caption("PhoBERT bị đóng băng (frozen) — chỉ BiLSTM + FC được train.")
