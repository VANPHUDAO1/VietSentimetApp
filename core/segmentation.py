"""VnCoreNLP segmentation and environment checks."""

import shutil
import streamlit as st
from functools import lru_cache
import threading

_segmenter_lock = threading.Lock()

from config.settings import VNCORENLP_DIR


class EnvironmentError_VnCoreNLP(RuntimeError):
    """Raised when the VnCoreNLP environment is not available."""


def check_java_available() -> bool:
    """Return True if Java is available on the system PATH."""
    return shutil.which("java") is not None


def check_environment() -> tuple[bool, str]:
    """Check Java and VnCoreNLP availability before app runtime."""
    if not check_java_available():
        return False, (
            "Không tìm thấy Java (JDK) trên máy. VnCoreNLP cần Java 8+ để "
            "chạy word segmentation. Cài JDK rồi khởi động lại app."
        )
    if not VNCORENLP_DIR.exists():
        return False, (
            f"Không tìm thấy thư mục VnCoreNLP tại {VNCORENLP_DIR}. "
            "Tải VnCoreNLP-1.2.jar + models/ và đặt đúng đường dẫn."
        )
    return True, ""


@st.cache_resource(show_spinner=False)
def get_segmenter():
    """Lazy-load the VnCoreNLP segmenter and raise friendly errors if unavailable."""
    if not check_java_available():
        raise EnvironmentError_VnCoreNLP(
            "Không tìm thấy Java (JDK) trên máy. VnCoreNLP cần Java 8+ để chạy. "
            "Vui lòng cài JDK rồi thử lại."
        )

    if not VNCORENLP_DIR.exists():
        raise EnvironmentError_VnCoreNLP(
            f"Không tìm thấy thư mục VnCoreNLP tại {VNCORENLP_DIR}. "
            "Vui lòng tải VnCoreNLP-1.2.jar + models/ và đặt đúng đường dẫn."
        )

    with _segmenter_lock:
        print("========== CREATE SEGMENTER ==========")
        import py_vncorenlp

        try:
            segmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"], save_dir=str(VNCORENLP_DIR)
            )
        except Exception as e:
            if "already running" in str(e).lower():
                raise EnvironmentError_VnCoreNLP(
                    "JVM đã được khởi tạo trước đó trong process này nhưng bước "
                    "load VnCoreNLP bị lỗi giữa chừng... Vui lòng dừng hẳn Streamlit "
                    "(Ctrl+C) rồi chạy lại `streamlit run app.py`."
                ) from e
            raise
        print("========== CREATED ==========")
        return segmenter


def segment_text(text: str) -> str:
    """Segment Vietnamese text into tokenized form."""
    segmenter = get_segmenter()
    sentences = segmenter.word_segment(text)
    return " ".join(sentences)
