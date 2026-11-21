import sys
from types import SimpleNamespace

import numpy as np
import pytest

import anonymize
import hardware


def fake_torch(built=True, available=True):
    return SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(
        is_built=lambda: built, is_available=lambda: available)),
        set_num_threads=lambda n: None, get_num_interop_threads=lambda: 1,
        mps=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None))


@pytest.mark.parametrize("system,architecture,built,available,expected", [
    ("Linux", "x86_64", True, True, "cpu"),
    ("Darwin", "x86_64", True, True, "cpu"),
    ("Darwin", "arm64", False, False, "cpu"),
    ("Darwin", "arm64", True, False, "cpu"),
    ("Darwin", "arm64", True, True, "mps"),
    ("Darwin", "aarch64", True, True, "mps"),
])
def test_native_device_selection(monkeypatch, system, architecture, built, available, expected):
    monkeypatch.setattr(hardware.platform, "system", lambda: system)
    monkeypatch.setattr(hardware.platform, "machine", lambda: architecture)
    torch = fake_torch(built, available)
    assert hardware.select_device("auto", torch) == expected
    assert hardware.select_device("cpu", torch) == "cpu"
    if expected == "cpu":
        with pytest.raises(ValueError, match="MPS requires"):
            hardware.select_device("mps", torch)
    else:
        assert hardware.select_device("mps", torch) == "mps"


def test_absent_mps_backend_and_unknown_device():
    torch = SimpleNamespace(backends=SimpleNamespace())
    assert hardware.mps_status(torch) == {"built": False, "available": False}
    with pytest.raises(ValueError, match="Device must"):
        hardware.select_device("cuda", torch)


def test_peak_memory_units(monkeypatch):
    monkeypatch.setattr(hardware.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1048576))
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    assert hardware.peak_rss_mb() == 1
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    assert hardware.peak_rss_mb() == 1024


def install_fake_modules(monkeypatch, reader):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=reader))
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")


def test_ocr_initialization_auto_falls_back_but_explicit_mps_aborts(monkeypatch):
    calls = []
    def reader(languages, gpu, **kwargs):
        calls.append(gpu)
        if gpu == "mps":
            raise RuntimeError("private model detail")
        return SimpleNamespace(device="cpu")
    install_fake_modules(monkeypatch, reader)
    ocr = anonymize.OCR(2, ["en"], device="auto")
    assert ocr.device == "cpu" and ocr.fallbacks == 1 and not ocr.mps_tested
    assert calls == ["mps", False]
    with pytest.raises(RuntimeError, match="initialization failed on mps"):
        anonymize.OCR(2, ["en"], device="mps")


def test_cpu_initialization_failure_is_reported_without_model_payload(monkeypatch):
    def reader(*args, **kwargs):
        raise RuntimeError("private model detail")
    install_fake_modules(monkeypatch, reader)
    with pytest.raises(RuntimeError, match="initialization failed on cpu") as error:
        anonymize.OCR(2, ["en"], device="cpu")
    assert "private" not in str(error.value)


def test_mps_models_must_really_reside_on_mps(monkeypatch):
    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type="cpu"))])
    install_fake_modules(monkeypatch, lambda *args, **kwargs: SimpleNamespace(device="mps", detector=model, recognizer=model))
    with pytest.raises(RuntimeError, match="initialization failed on mps"):
        anonymize.OCR(2, ["en"], device="mps")


def test_mps_inference_failure_retries_detection_and_recognition_on_cpu(monkeypatch):
    calls = []
    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type="mps"))])
    class Reader:
        def __init__(self, languages, gpu, **kwargs):
            self.device = "mps" if gpu else "cpu"
            self.detector = self.recognizer = model
        def detect(self, gray, **kwargs):
            calls.append(("detect", self.device))
            return [[[1, 20, 1, 10]]], [[]]
        def recognize(self, *args, **kwargs):
            calls.append(("recognize", self.device))
            if self.device == "mps":
                raise RuntimeError("unsupported operation")
            return [(np.float32([[1, 1], [20, 1], [20, 10], [1, 10]]), "SAMPLE", .9)]
    install_fake_modules(monkeypatch, Reader)
    monkeypatch.setattr(anonymize.OCR, "smoke_test", lambda self: True)
    ocr = anonymize.OCR(2, ["en"], device="auto")
    results = ocr.read(np.zeros((32, 64, 3), np.uint8))
    assert len(results) == 1 and ocr.calls == 1 and ocr.fallbacks == 1
    assert calls == [("detect", "mps"), ("recognize", "mps"), ("detect", "cpu"), ("recognize", "cpu")]
    assert ocr.device == "cpu"
    calls.clear()
    explicit = anonymize.OCR(2, ["en"], device="mps")
    with pytest.raises(RuntimeError, match="MPS OCR inference failed"):
        explicit.read(np.zeros((32, 64, 3), np.uint8))
    assert calls == [("detect", "mps"), ("recognize", "mps")]


def test_run_always_closes_ocr_after_processing_error(monkeypatch):
    closed = []
    def fail(args, resources):
        resources.append(SimpleNamespace(close=lambda: closed.append(True)))
        raise RuntimeError("stop")
    monkeypatch.setattr(anonymize, "_run", fail)
    with pytest.raises(RuntimeError, match="stop"):
        anonymize.run(None)
    assert closed == [True]


def test_hardware_diagnostics_distinguishes_detection_from_inference(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(False, False))
    info = hardware.hardware_info("cpu")
    assert info["selected_device"] == "cpu"
    assert not info["ocr_pipeline_tested"] and not info["mps_pipeline_tested"]
    assert not info["mps"]["built"]


def test_failed_cpu_fallback_aborts_without_results(monkeypatch):
    def reader(*args, **kwargs):
        raise RuntimeError("private failure details")
    install_fake_modules(monkeypatch, reader)
    with pytest.raises(hardware.OCRBackendError, match="CPU OCR initialization failed after MPS failure"):
        anonymize.OCR(2, ["en"], device="auto")


def test_mps_model_wrappers_are_removed_and_smoke_test_required(monkeypatch):
    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type="mps"))])
    wrapper = SimpleNamespace(module=model)
    install_fake_modules(monkeypatch, lambda *a, **kw: SimpleNamespace(device="mps", detector=wrapper, recognizer=wrapper))
    tested = []
    monkeypatch.setattr(anonymize.OCR, "smoke_test", lambda self: tested.append(self.device))
    ocr = anonymize.OCR(2, ["en"], device="mps")
    assert tested == ["mps"] and ocr.mps_tested
    assert ocr.reader.detector is model and ocr.reader.recognizer is model
    ocr.close()
    assert ocr.reader is None


def test_failed_mps_smoke_test_falls_back_without_claiming_gpu_success(monkeypatch):
    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type="mps"))])
    def reader(*args, gpu, **kwargs):
        return SimpleNamespace(device="mps" if gpu else "cpu", detector=model, recognizer=model)
    def fail(self):
        raise RuntimeError("unreadable synthetic sample")
    install_fake_modules(monkeypatch, reader)
    monkeypatch.setattr(anonymize.OCR, "smoke_test", fail)
    ocr = anonymize.OCR(2, ["en"], device="auto")
    assert ocr.device == "cpu" and not ocr.mps_tested and ocr.fallbacks == 1


def test_hardware_info_does_not_process_video(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["anonymize.py", "--hardware-info"])
    monkeypatch.setattr(anonymize, "hardware_info", lambda requested: {"selected_device": "cpu"})
    monkeypatch.setattr(anonymize, "run", lambda args: pytest.fail("Video processing must not start"))
    assert anonymize.main() == 0
    assert '"selected_device": "cpu"' in capsys.readouterr().out
