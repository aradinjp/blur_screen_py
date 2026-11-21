"""Reproducible local benchmark; outputs contain geometry and IDs, never OCR text.

Run each configuration in a fresh process to avoid shared thread pools/RSS.
Baseline reproduces the original BGR readtext + exact matching + mp4v/blur loop.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from anonymize import Matcher, OCR, ocr_masks
from hardware import peak_rss_mb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["baseline", "variants"], default="baseline")
    p.add_argument("--input", default="recording.mp4")
    p.add_argument("--config", default="redaction_data.json")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--opencv-threads", type=int, default=1)
    p.add_argument("--frames", default="0,40,50,141,164", help="comma separated or all")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cv2.setNumThreads(args.opencv_threads)
    config = json.loads(Path(args.config).read_text()); matcher = Matcher(config["frazy_do_ukrycia"])
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    start = time.perf_counter(); t = time.perf_counter(); ocr = OCR(args.threads, ["en"])
    init = time.perf_counter() - t
    cap = cv2.VideoCapture(args.input); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = list(range(count)) if args.frames == "all" else [int(i) for i in args.frames.split(",")]
    rows = []; decode_s = encode_s = 0
    with tempfile.TemporaryDirectory(prefix="blur-benchmark-") as tmp:
        writer = cv2.VideoWriter(str(Path(tmp) / "baseline.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
        if not writer.isOpened():
            raise RuntimeError("Cannot initialize baseline encoder")
        try:
            for idx in frame_ids:
                t = time.perf_counter()
                if args.frames != "all":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read(); decode_s += time.perf_counter() - t
                if not ok:
                    raise RuntimeError("Cannot decode input")
                if args.kind == "baseline":
                    t = time.perf_counter(); horizontal, free = ocr.reader.detect(frame)
                    detector = time.perf_counter() - t
                    t = time.perf_counter()
                    results = ocr.reader.recognize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), horizontal[0], free[0])
                    recognizer = time.perf_counter() - t
                    exact = []
                    t = time.perf_counter()
                    for poly, text, prob in results:
                        ids = [i for i, phrase in enumerate(config["frazy_do_ukrycia"]) if phrase.upper() in text.upper()]
                        if ids:
                            x1, y1 = np.maximum(0, np.asarray(poly[0], int) - 5)
                            x2, y2 = np.minimum([frame.shape[1], frame.shape[0]], np.asarray(poly[2], int) + 5)
                            if x2 > x1 and y2 > y1:
                                frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (51, 51), 30)
                                exact.append({"phrase_ids": ids, "polygon": np.asarray(poly).tolist()})
                    writer.write(frame); encode_s += time.perf_counter() - t
                    enhanced = ocr_masks(results, matcher)
                    rows.append({"frame": idx, "detector_s": detector, "recognizer_s": recognizer,
                                 "detected_boxes": len(results), "exact_masks": exact,
                                 "enhanced_matching_masks": [m.record() for m in enhanced]})
                else:
                    y0 = frame.shape[0] // 2
                    roi = frame[y0:]; variants = [("raw1", 1, "none"), ("raw2", 2, "none"),
                        ("raw3", 3, "none"), ("clahe2", 2, "clahe"),
                        ("threshold2", 2, "threshold"), ("sharpen2", 2, "sharpen"),
                        ("rotation2", 2, "none")]
                    for name, scale, preprocess in variants:
                        d0, r0 = ocr.detect_s, ocr.recognize_s
                        results = ocr.read(roi, scale, preprocess, rotation=name == "rotation2")
                        matches = ocr_masks(results, matcher)
                        rows.append({"frame": idx, "variant": name, "detector_s": ocr.detect_s - d0,
                                     "recognizer_s": ocr.recognize_s - r0, "boxes": len(results),
                                     "matches": [m.record() for m in matches]})
                if idx % 25 == 0 or idx == frame_ids[-1]:
                    print(f"Benchmark {args.kind}: {idx}/{count}, {time.perf_counter() - start:.1f} s", flush=True)
        finally:
            writer.release(); cap.release()
    total = time.perf_counter() - start
    report = {"threads": args.threads, "opencv_threads": args.opencv_threads, "init_s": init,
              "elapsed_s": total, "decode_s": decode_s, "encode_and_blur_s": encode_s,
              "rss_mb": peak_rss_mb(),
              "frames_analyzed": len(frame_ids), "kind": args.kind, "frames": rows}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"Finished: {len(frame_ids)} frames, {total:.2f} s, RSS {report['rss_mb']:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
