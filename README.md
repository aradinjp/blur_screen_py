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
without re-encoding. Inputs without audio are supported. Variable frame rate and
damaged streams are rejected explicitly to prevent output with an incorrect
duration or frame count. Reduced grayscale frames are buffered on disk, while
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
