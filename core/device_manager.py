"""Device selection and GPU availability handling."""

from dataclasses import dataclass

import platform
import multiprocessing

import torch


@dataclass
class DeviceResolution:
    requested: str  # "cpu" hoặc "cuda" — người dùng chọn
    resolved: str  # thiết bị THỰC TẾ sẽ dùng sau khi kiểm tra
    fallback_happened: bool  # True nếu resolved != requested
    message: str = ""


def is_cuda_available() -> bool:
    """Return True when CUDA is available."""
    return torch.cuda.is_available()


def resolve_device(user_choice: str) -> DeviceResolution:
    """Resolve the requested compute device, with fallback to CPU."""
    requested = "cuda" if user_choice.lower() == "gpu" else "cpu"

    if requested == "cuda" and not is_cuda_available():
        return DeviceResolution(
            requested=requested,
            resolved="cpu",
            fallback_happened=True,
            message="Máy không có GPU (CUDA) khả dụng — đã tự động chuyển về CPU.",
        )

    return DeviceResolution(
        requested=requested, resolved=requested, fallback_happened=False
    )


def get_device_display_info(user_choice: str) -> str:
    """Return a friendly description for the selected device option."""
    normalized_choice = user_choice.lower()

    if normalized_choice == "gpu":
        if not is_cuda_available():
            return "GPU không khả dụng trên máy này"

        device_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"GPU — {device_name} • {total_mem_gb:.1f} GB VRAM"

    cpu_name = platform.processor() or platform.machine() or "Không xác định"
    cpu_cores = multiprocessing.cpu_count() or 1
    return (
        f"CPU — {cpu_name} • {cpu_cores} lõi • "
        f"{max(1, cpu_cores // 2)} luồng hiệu quả"
    )
