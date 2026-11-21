# Blurring JSON-specified data in screen recordings

The default output preserves the appearance of the original program: **blurred,
narrow strips over sensitive data**, a 5 px margin, and no full-card masking.
Phrases come exclusively from `redaction_data.json`. `anonymize.py` is the
primary entry point. `legacy_anonymize.py` and `blur_json.py` are compatibility
entry points and run the same program.

The program contains no fixed field coordinates, interface colors, or card layout.
OCR searches the entire frame, allowing it to process recordings with different
layouts. Detected strips are tracked in both temporal directions. During strong
perspective changes, a surface homography is validated as an additional aid; only
text strips are transferred. A data layout is never transferred from one side of
a rotating card to the other.

Install on Arch Linux / WSL2 x86_64:

```bash
sudo pacman -S --needed python python-pip ffmpeg
python -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip check
```

The verified environment uses Python 3.14.10, EasyOCR 1.7.2, torch 2.15.0+cpu,
torchvision 0.29.2+cpu, NumPy 2.5.5, and OpenCV headless 5.1.0.95. Python 3.12
or newer is required; other interpreter versions need separate verification. The
CPU environment contains no NVIDIA/CUDA packages or second OpenCV variant.
torchvision remains because EasyOCR requires it. `requirements.txt` references
the CPU dependency set, and pip installs transitive dependencies. The tested
versions are recorded in `diagnostics/environment_cpu.txt`. Sources:
[EasyOCR](https://github.com/JaidedAI/EasyOCR/blob/v1.7.2/requirements.txt),
[torch CPU](https://download.pytorch.org/whl/cpu/torch/), and
[torchvision CPU](https://download.pytorch.org/whl/cpu/torchvision/).

```bash
python anonymize.py \
  --input recording.mp4 \
  --output recording_redacted_v2.mp4 \
  --config redaction_data.json \
  --mode blur \
  --threads 6
```

On the first run, add `--download-models` if the EasyOCR models are not already
in the local cache (`~/.EasyOCR/model/`). Without this option, the program does
not download models. Existing output is not overwritten unless `--overwrite` is
provided. The source video and JSON configuration cannot be selected as output
or report destinations.

| Setting | Default | Meaning |
|---|---:|---|
| `--mode` | `blur` | Blur; `pixelate` and `solid` are also available |
| `--margin` | 5 | Margin around the text polygon in input pixels |
| `--blur-sigma` | 30 | Blur with a 51 × 51 kernel, matching the original version |
| `--threads` | 6 | PyTorch CPU threads |
| `--opencv-threads` | 1 | Separate OpenCV thread pool |
| `--ocr-every` | 5 | Run OCR every N frames, after scene changes, and on the last frame |
| `--hold-frames` | 10 | Maximum tracking distance in each temporal direction |
| `--languages` | `en` | For example `pl,en`, depending on the text and installed models |
| `--crf` | 18 | H.264/libx264 quality |

Tracking requires forward-backward point agreement and RANSAC geometry. Surface
homographies also undergo outline and post-transform image checks. A transform
is rejected when these checks disagree. Normalization and bounded fuzzy matching
handle spacing, month abbreviations, `0/O`, `1/I/l`, `8/B`, and dates split into
adjacent fragments. The smallest matching row segment is selected to avoid
blurring neighboring labels.

The `<output>.report.json` report records stage timings, RSS, mask geometry, and
phrase indices without recognized personal data. Frame indices are **zero-based**.
`tracking_rejection_frames` lists frames with rejected tracking attempts. Such an
attempt does not always mean that a mask is absent because a valid observation may
exist from the opposite direction. `uncertainty_frames` counts these frames for
review.

```bash
python anonymize.py --output preview.mp4 --diagnostics diagnostics/masks \
  --diagnostic-frames 40,41,42,43,50,52,53,54,141,142,143,164,165,166
```

Diagnostic PNG files show the blurred image with strip outlines. H.264 output
preserves resolution, frame count, and constant frame rate; audio is stream-copied
without re-encoding. Inputs without audio are supported. Variable frame rate inputs
are automatically converted to a temporary constant-FPS copy at the average input
frame rate before OCR. This may duplicate or drop frames while preserving playback
timing to within a frame. The temporary video uses lossless H.264 and is removed
after processing; allow additional disk space. Reports include original input and
processing video parameters. Damaged streams are rejected explicitly. Reduced grayscale frames are buffered on disk, while
full-resolution frames are not retained in RAM together.

Optional `regions` in JSON can restrict OCR to selected areas, but they are not
required and are not used by the standard data set. `bounds: [x,y,width,height]`
uses relative coordinates from 0 to 1 with `policy: "ocr"` and `scale: 2`. If
provided, regions must cover every possible location of sensitive data. Optional
`--fail-closed` masks an entire recognized moving object; **it is disabled by
default and was not used in the corrected strip-based video**.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tools/test_other_layouts.py
python tools/verify_video.py
```

The additional-layout test uses real EasyOCR and phrases from JSON with two color
palettes and different text positions. It requires a font available through
`fc-match`. Coordinates in tests and verification tools describe only test media;
the anonymization program does not read them. Benchmarks are in
`tools/benchmark.py` and `tools/benchmark_tracking.py`, with measured results in
`REPORT.md`.

Local comparisons in `diagnostics/verification/` include original image content
and require the same protection as the source recording. OCR output and coverage
of OCR rectangles alone do not prove complete anonymization. Completely missed
text, very low contrast, or a brief appearance between OCR samples can remain
unmasked. `--ocr-every 1` increases analysis frequency. Blur preserves the chosen
appearance but does not guarantee irreversibility. Perform a full-resolution human
review before publication, especially for new recordings.

## macOS Apple Silicon Installation

Use a native ARM64 shell and Python 3.12 on an M-series Mac. This dependency set
targets macOS 14 or newer; MPS must also be supported by the installed PyTorch
build and operating system. Rosetta is not required. On a Mac with Homebrew:

```bash
brew install python@3.12 ffmpeg
./setup_macos.sh --download-models
source .venv-macos/bin/activate
```

The script does not install Homebrew, change system configuration, or invoke sudo.
Omit `--download-models` when models are already in `~/.EasyOCR/model/`. Downloads
are opt-in; the installation fails clearly if models are missing. Set `PYTHON_BIN`
to an absolute path to native Python 3.12 if it is not on PATH. The dedicated
`.venv-macos` uses `requirements-macos.txt`; do not install the Linux CPU
requirements into it. For tests, install pytest separately:

```bash
python -m pip install pytest==9.1.1
python -m pytest -q
```

The macOS dependency set pins torch 2.6.0 with torchvision 0.21.0, EasyOCR 1.7.2,
NumPy 2.2.6, OpenCV headless 4.11.0.86, Pillow 11.2.1, SciPy 1.15.3, and
scikit-image 0.25.2. Their published Python 3.12 ARM64 wheel metadata and dependency
constraints were checked. Installation and inference with this set have not been
tested on a physical Mac. Transitive dependencies are resolved by pip, followed
by `pip check`. Only headless OpenCV is installed; no CUDA packages are needed on
macOS. Linux dependency files are unchanged.

```bash
python anonymize.py --input video.mp4 --output video_anonymized.mp4 \
  --config redaction_data.json --mode blur --device auto
python anonymize.py --input video.mp4 --output video_cpu.mp4 --device cpu
python anonymize.py --input video.mp4 --output video_mps.mp4 --device mps
```

`auto` selects MPS only on native ARM64 macOS when both
`torch.backends.mps.is_built()` and `is_available()` succeed. Initialization checks
the actual detector and recognizer parameter devices, then runs detection and
recognition of synthetic text before accepting MPS. Linux/WSL2 `auto` retains CPU
execution. Explicit `mps` fails if unavailable or initialization fails. In `auto`,
an MPS exception discards the entire affected OCR result, initializes CPU models,
and repeats detection and recognition for the same original image. A failed CPU
retry aborts processing; no partially processed output is committed. Remaining OCR
calls use CPU. Valid earlier observations and all temporal safeguards are retained.

The historical `cenzura.py` launcher also works with the same options. Previously
renamed input and configuration files retain their English names. An existing
configuration with a different filename can be selected using `--config`; its
`frazy_do_ukrycia` key and values require no conversion.

Hardware diagnostics use a separate flag so the existing `--diagnostics DIRECTORY`
option continues saving mask previews:

```bash
python anonymize.py --hardware-info --device auto
python anonymize.py --hardware-info --inference-test --device mps
```

The first command reports detection only. In the second command, require
`selected_device: "mps"`, `mps.built: true`, `mps.available: true`,
`ocr_pipeline_tested: true`, and `mps_pipeline_tested: true` to confirm the OCR
smoke test ran on MPS. The video report also records the final device, requested
device, MPS smoke-test status, and CPU fallback count. This demonstrates execution,
not equivalent accuracy on all recordings; review masks on representative footage.

MPS accelerates PyTorch model inference on the GPU. OpenCV preprocessing, optical
flow, homographies, mask rasterization, and encoding remain on the existing CPU
path. Frames are still buffered as reduced grayscale images on disk. Recognition
keeps the existing batch size and precision: there is no measured evidence here
to justify changing them. Model resources and the MPS allocator cache are released
after processing, including errors. RSS is reported in MiB with macOS-specific
units; it does not separately measure Metal allocations or total unified memory.

MPS does not use the Apple Neural Engine. Apple Vision/Core ML require a separate
backend and accuracy validation. They have not been added without a demonstrated
benefit. H.264/libx264 CRF 18 and stream-copied audio remain the default. Apple's
VideoToolbox encoder is not enabled because its quality and speed have not been
measured on this project.

To benchmark on the Mac, cache models first, then use a new output directory:

```bash
python anonymize.py --hardware-info --inference-test --device cpu
python anonymize.py --hardware-info --inference-test --device mps
python tools/benchmark_devices.py --devices cpu,mps --repeats 3 \
  --input recording.mp4 --config redaction_data.json \
  --output-dir diagnostics/mac_device_benchmark
```

Each device/repeat runs in a fresh process. The benchmark measures initialization,
a synthetic warm-up, representative OCR calls, and a separate complete video
processing run with the same masking, thread count, OCR frequency, and encoding
settings. Model downloads are disabled. MPS timings synchronize the device. Reports
include per-frame OCR times, total runtime, effective FPS, peak process RSS, output
metadata validation, unchanged audio packet hashes, and CPU/MPS mask bitmap
differences for every frame. `--samples-only` skips full video generation for a
quick timing check. `--frames` accepts comma-separated zero-based sample indices;
choose valid indices for other videos. CPU masks are a comparison baseline rather
than ground truth; inspect all changed frames and the final videos visually.
Do not publish accelerated output solely because the benchmark completed.

MPS may be slower for small images or when CPU tracking/encoding dominates. No Mac
speedup is claimed, and `auto` does not run a latency calibration. Use `--device cpu`
when measurements favor CPU. Common troubleshooting:

- MPS built but unavailable: check native architecture, macOS support, and the
  installed PyTorch build. Hardware presence alone is insufficient.
- Missing models: rerun setup with `--download-models` or provide the local cache.
- Unsupported operation, device mismatch, or out-of-memory: `auto` retries the
  complete OCR operation on CPU; explicit `mps` aborts. Test `--device cpu` directly.
- Do not blindly enable `PYTORCH_ENABLE_MPS_FALLBACK=1`. It can move unsupported
  operators to CPU, add transfers, and obscure performance; diagnostics report its
  value. It is never enabled by this application and cannot fix all errors.
- Do not disable MPS memory limits or enable reduced precision to make a benchmark
  pass. Quit other large applications or select CPU on smaller-memory Macs.

See `APPLE_SILICON_REPORT.md` for executed tests and the remaining hardware checks.
