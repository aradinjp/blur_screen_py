# Analysis and test results

The finished video is `recording_redacted_blur.mp4`, an identical copy of the final
`recording_redacted_v2.mp4`. It uses **narrow blurred strips without covering the
entire card**. Masks cover 4.34% of a frame on average and 4.91% at most. Default
settings are `blur`, a 5 px margin, a 51 × 51 kernel, and sigma 30. Automatic
full-object masking is disabled.

Missing masks were caused by independent per-frame analysis, literal phrase
matching within one OCR result, and loss of small-text readability under
perspective distortion. Split dates and minor character errors previously did not
produce a mask. Using only OCR-box vertices 0 and 2 did not correctly describe
every tilted rectangle. OpenCV VideoWriter also wrote video without audio.

A separate OCR-stage measurement in `diagnostics/ocr_stages.json` distinguishes
the causes. On frame 40, the detector covered the date, but recognized text did
not match JSON. Contrast adjustment alone did not help; adjusted read geometry
and normalization recovered the match. On frame 50, the detector covered only
about 53% of the date field, while the name and expiry date had detection regions
but incorrect recognition. On frame 141, the original detector missed three small
fields; the tuned detector found them but still did not recognize their values.
On frame 164, regions were detected but no reading matched JSON. High confidence
for one incorrect fragment does not prove that the complete field was recognized,
so changing OCR thresholds alone is insufficient.

The implementation adds normalization, bounded fuzzy matching, joining of short
row fragments, and selection of the smallest matching area. Masks are propagated
between adjacent frames in both directions. During strong rotation, an auxiliary
surface outline initializes rectification and point tracking; point agreement,
RANSAC, outline agreement, and image agreement validate the homography. Only text
strips are propagated. The remaining card surface, portrait, and interface stay
visible.

The program contains no field coordinates, color template, or layout for a
specific card. Every phrase is read from the unchanged JSON file. Coordinates in
test files are used only to verify the sample recording.

The full test processed all 261 source frames and compared the result with the
previous output. The new file is 1030 × 1102, 30 FPS, H.264/yuv420p, 261 frames,
and 8.700000 seconds long. The difference from the 8.699967-second input metadata
is 33 microseconds. AAC was preserved without re-encoding. The SHA256 of raw audio
packets is identical before and after processing:
`50b66bba53deaca6a7a0b45e84f1e395e986bb484a9fd983213dd1a746823db9`.
The source and previous output were not modified.

| Final-output check | Measured result |
|---|---:|
| Frames analyzed | 261 |
| Frames processed by OCR | 53 |
| Frames with masks | 261 |
| Frames with strips on the card | 258 |
| Regions detected by OCR | 416 |
| Mask records, including tracking | 2901 |
| Rejected tracking chains | 40 |
| Frames with rejected attempts requiring review | 50, 51, 142, 166 |
| Static-panel field checks | 1044, no detected omission |
| Annotated card-field checks | 48, no detected omission |
| PSNR outside masks and their surroundings | 37.90 dB |

Mask records can overlap; their count does not represent people or distinct
fields. A rejected attempt does not mean exposed data: frame 50 received valid
masks from other observations. On frames 51, 142, and 166, the card is nearly
edge-on and the review found no readable values.

| Frames, zero-based | Previous output | Final output |
|---|---|---|
| 40, 41, 42, 43 | Readable date of birth | Date and remaining fields blurred with strips |
| 50 | Some strips missing during rotation | Four data regions reconstructed |
| 52 | Mostly the name mask; other fields readable | Strips cover all four annotated fields |
| 53 | Some small-field masks missing | Fields covered by strips |
| 54 | Basic masks present | Field blur preserved |
| 141 | Visible data under strong perspective shortening | Narrow masks propagated by homography |
| 142 | Card almost entirely edge-on | No readable values found during review |
| 143 | Missing masks on the other side after rotation | Corresponding fields on that side blurred |
| 164, 165 | Visible data during another rotation | Four data regions blurred |
| 166 | Card almost entirely edge-on | No readable values found during review |

Coverage of all 2029 earlier OCR rectangles was checked. Two rectangles on frames
52 and 167 contain different extreme background pixels at their borders than the
new strips. Their cores are fully covered after excluding a two-pixel border.
Both cases were reviewed at increased scale, and no exposed letters were found.
The raw differences remain in the verification output.

Contact sheets for all 261 frames and enlarged views of the specified moments were
reviewed. Original / previous / new comparisons are stored locally in
`diagnostics/verification/`; PNG files with new strip outlines are in
`diagnostics/masks_blur/`. Geometry and detail-reduction checks for annotated
fields supplement visual review; OCR output alone is not evidence of safety.

CPU benchmark with 6 PyTorch threads and 1 OpenCV thread:

| Stage | Original rules, OCR every frame | Final strip-based version |
|---|---:|---:|
| OCR initialization and import | 3.48 s | 3.40 s |
| OCR detector | 787.12 s | 166.79 s |
| Character recognition | 261.13 s | 47.53 s |
| Auxiliary outline detection | — | 1.16 s |
| Tracking | — | 9.58 s |
| Encoding and final validation | — | 5.07 s |
| Total | 1061.55 s | 236.68 s |
| Average per frame | 4.07 s | 0.907 s |
| Peak process RSS | 1932 MiB | 1611 MiB |

This recording processed about **4.5× faster**. The old benchmark reproduces the
original detector, recognizer, literal matching, blur, and mp4v path, and includes
about 7.5 seconds of extra geometry instrumentation. The original environment had
a torch build with CUDA libraries while running with `gpu=False`. The final run
used a fresh virtual environment with `+cpu` packages and one OpenCV variant.

Thread-count measurements on the same five frames with default OCR:

| PyTorch threads | Average detector and recognizer time |
|---:|---:|
| 1 | 12.328 s/frame |
| 3 | 5.382 s/frame |
| 6 | 3.527 s/frame |
| 11 | 3.883 s/frame |

An auxiliary tracking measurement using an earlier geometry cache took 14.02,
11.33, and 11.19 seconds with 1, 3, and 6 OpenCV threads respectively. This is a
separate measurement, not the complete final algorithm benchmark. Defaults remain
1 OpenCV thread and 6 PyTorch threads. The 22-thread attempt was interrupted before
the first frame completed and is not presented as a completed benchmark.

Four difficult cropped frames were also tested at 1× and 2× raw scale, 3×, with
CLAHE, thresholding, sharpening, and OCR rotations. Average times were 1.72, 5.81,
9.07, 5.94, 6.16, 6.06, and 8.04 seconds per crop respectively. Scaling to 2×
recovered one additional number on frame 50; thresholding recovered one additional
number on frame 164. No option consistently improved every field, so these costly
options remain disabled by default. Temporal consistency and geometry provide the
main improvement.

`pytest`: **22 tests passed** in the CPU environment. Coverage includes
normalization, fragmented dates, OCR confusions, minimal strip extent, frame
boundaries, homography, detection gaps, scene changes, invalid inputs, cleanup
after exceptions, protection of the source and its hard link, VFR rejection, and
real encoding with and without AAC.

A separate real-EasyOCR test used JSON phrases in a synthetic 800 × 450 video with
two unrelated layouts, dark and light backgrounds, different text positions and
colors, and a scene change. It completed **6 frames, 24 value checks, and full
letter coverage in annotated fields**, with at most 11.38% of a frame covered by
strips and no full-object mask. Output and preview files are in
`diagnostics/other_layouts/`. This validates two variants rather than every
possible interface.

No exposed values were found in the completed annotated-field review. This is not
a claim of 100% effectiveness. Poorly readable text may never be recognized, a
brief appearance can fall between OCR samples, and tracking can fail under
occlusion or loss of geometry. New recordings require full-resolution human
review. `--ocr-every 1` increases analysis frequency. Blur does not guarantee
irreversibility, and the preserved audio track is not anonymized.

Run another recording with:

```bash
source .venv-cpu/bin/activate
python anonymize.py --input another_recording.mp4 --output blurred_output.mp4 \
  --config redaction_data.json --mode blur --threads 6
```
