"""Overlay popup for the 4-step sentiment analysis pipeline."""

import html
import time
from enum import Enum

import streamlit as st

from config.settings import POPUP_RENDER_FLUSH_SEC, POPUP_SUCCESS_CLOSE_SEC

STEP_LABELS = [
    "Kiểm tra đầu vào",
    "Tải model",
    "Tiền xử lý",
    "Suy luận",
]


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


def _step_line(index: int, label: str, status: StepStatus, detail: str = "") -> str:
    icons = {
        StepStatus.PENDING: f"{index + 1}.",
        StepStatus.RUNNING: "⏳",
        StepStatus.DONE: "✅",
        StepStatus.ERROR: "❌",
    }
    colors = {
        StepStatus.PENDING: "#94a3b8",
        StepStatus.RUNNING: "#1d4ed8",
        StepStatus.DONE: "#15803d",
        StepStatus.ERROR: "#dc2626",
    }
    margin_top = "10px" if index == 0 else "6px"
    weight = "700" if status == StepStatus.RUNNING else "400"
    detail_html = (
        f' <span style="color:#64748b;font-size:0.85rem;">({html.escape(detail)})</span>'
        if detail
        else ""
    )
    return (
        f'<div style="margin-top:{margin_top};color:{colors[status]};font-weight:{weight};">'
        f"{icons[status]} {html.escape(label)}{detail_html}</div>"
    )


def _render_popup_html(
    steps: list[tuple[str, StepStatus, str]],
    title: str,
    subtitle: str = "",
) -> str:
    progress_done = sum(1 for _, status, _ in steps if status == StepStatus.DONE)
    progress_pct = int(progress_done / len(steps) * 100) if steps else 0
    steps_html = "".join(
        _step_line(index, label, status, detail)
        for index, (label, status, detail) in enumerate(steps)
    )
    subtitle_html = (
        f'<div style="font-size:0.88rem;color:#64748b;margin-top:6px;">'
        f"{html.escape(subtitle)}</div>"
        if subtitle
        else ""
    )

    return f"""
    <div style="position:fixed;top:0;left:0;width:100%;height:100%;
        background:rgba(15,23,42,0.45);z-index:9999;display:flex;
        align-items:center;justify-content:center;">
        <div style="background:white;padding:24px 28px;border-radius:16px;
            box-shadow:0 12px 30px rgba(0,0,0,0.2);min-width:360px;max-width:420px;">
            <div style="font-size:1.05rem;font-weight:700;color:#1d4ed8;">
                {html.escape(title)}
            </div>
            {subtitle_html}
            <div style="margin-top:12px;height:6px;background:#e2e8f0;
                border-radius:999px;overflow:hidden;">
                <div style="width:{progress_pct}%;height:100%;
                    background:linear-gradient(90deg,#4f8cff,#7c4dff);"></div>
            </div>
            <div style="margin-top:14px;">{steps_html}</div>
        </div>
    </div>
    """


def _render_success_html(detail: str) -> str:
    return f"""
    <div style="position:fixed;top:0;left:0;width:100%;height:100%;
        background:rgba(15,23,42,0.45);z-index:9999;display:flex;
        align-items:center;justify-content:center;">
        <div style="background:white;padding:24px 28px;border-radius:16px;
            box-shadow:0 12px 30px rgba(0,0,0,0.2);min-width:360px;max-width:420px;">
            <div style="font-size:1.05rem;font-weight:700;color:#15803d;">✅ Thành công</div>
            <div style="margin-top:10px;color:#475569;">{html.escape(detail)}</div>
            <div style="margin-top:8px;font-size:0.85rem;color:#94a3b8;">
                Popup sẽ tự đóng sau vài giây...
            </div>
        </div>
    </div>
    """


class ProcessingPopup:
    """Manage the 4-step processing overlay shown during prediction."""

    def __init__(self, placeholder):
        self._placeholder = placeholder
        self._subtitle = ""
        self._steps = [(label, StepStatus.PENDING, "") for label in STEP_LABELS]

    def start(self, subtitle: str = "") -> None:
        self._subtitle = subtitle
        self._steps[0] = (STEP_LABELS[0], StepStatus.RUNNING, "")
        self._render(subtitle=subtitle)

    def complete_step(self, step_index: int, detail: str = "") -> None:
        self._steps[step_index] = (STEP_LABELS[step_index], StepStatus.DONE, detail)
        next_index = step_index + 1
        if next_index < len(self._steps):
            label = STEP_LABELS[next_index]
            self._steps[next_index] = (label, StepStatus.RUNNING, "")
        self._render()

    def show_success_and_close(
        self,
        detail: str = "Đã xử lý xong câu của bạn.",
        delay_sec: float = POPUP_SUCCESS_CLOSE_SEC,
    ) -> None:
        self._placeholder.markdown(_render_success_html(detail), unsafe_allow_html=True)
        time.sleep(delay_sec)
        self._placeholder.empty()

    def close(self) -> None:
        self._placeholder.empty()

    def _render(self, title: str = "⏳ Đang xử lý...", subtitle: str | None = None) -> None:
        if subtitle is None:
            subtitle = self._subtitle
        self._placeholder.markdown(
            _render_popup_html(self._steps, title, subtitle),
            unsafe_allow_html=True,
        )
        time.sleep(POPUP_RENDER_FLUSH_SEC)
