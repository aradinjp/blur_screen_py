import importlib.metadata
import os
import platform
import resource
import shutil


class OCRBackendError(RuntimeError):
    pass


def mps_status(torch):
    backend = getattr(torch.backends, "mps", None)
    return {"built": bool(backend and backend.is_built()),
            "available": bool(backend and backend.is_available())}


def select_device(requested, torch):
    if requested not in ("auto", "cpu", "mps"):
        raise ValueError("Device must be auto, cpu, or mps")
    if requested == "cpu":
        return "cpu"
    native = platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")
    status = mps_status(torch)
    supported = native and status["built"] and status["available"]
    if requested == "mps" and not supported:
        raise ValueError("MPS requires native ARM64 macOS and a PyTorch build with MPS built and available")
    return "mps" if supported else "cpu"


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024 if platform.system() == "Darwin" else 1024)


def hardware_info(requested="auto"):
    import torch
    versions = {}
    for package in ("torch", "torchvision", "easyocr", "numpy", "opencv-python-headless", "Pillow", "scikit-image"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"platform": platform.system(), "architecture": platform.machine(),
            "python": platform.python_version(), "dependencies": versions,
            "mps": mps_status(torch), "requested_device": requested,
            "selected_device": select_device(requested, torch), "ocr_backend": "EasyOCR",
            "ocr_pipeline_tested": False, "mps_pipeline_tested": False,
            "operator_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
            "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")}
