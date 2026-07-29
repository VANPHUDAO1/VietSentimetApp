"""Device selection and GPU availability handling."""

from dataclasses import dataclass

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
