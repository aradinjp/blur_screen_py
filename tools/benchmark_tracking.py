"""Compare OpenCV thread pools on this recording without any competing OCR."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from anonymize import Mask, object_candidates, temporal_masks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="recording.mp4")
    p.add_argument("--report", default="recording_redacted_v2.report.json")
    p.add_argument("--output", default="diagnostics/opencv_threads.json")
    args = p.parse_args()
    report = json.loads(Path(args.report).read_text()); meta = report["input"]
    fields = []; scenes = []
    for row in report["frames"]:
        fields.append([Mask(np.float32(m["polygon"])) for m in row["masks"]
                       if m["source"] in ("ocr", "ocr_joined")])
        scenes.append(row["scene"])
    results = []
    with tempfile.TemporaryDirectory(prefix="blur-flow-") as tmp:
        shape = (round(meta["height"] / 2), round(meta["width"] / 2))
        gray = np.memmap(Path(tmp) / "gray.bin", dtype=np.uint8, mode="w+", shape=(meta["frames"], *shape))
        cap = cv2.VideoCapture(args.input)
        try:
            for i in range(meta["frames"]):
                ok, f = cap.read(); assert ok
                gray[i] = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), shape[::-1])
        finally:
            cap.release()
        for threads in (1, 3, 6):
            cv2.setNumThreads(threads); cv2.setRNGSeed(0)
            t = time.perf_counter(); cap = cv2.VideoCapture(args.input); objects_count = 0
            try:
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    objects_count += len(object_candidates(f))
            finally:
                cap.release()
            detection_s = time.perf_counter() - t
            t = time.perf_counter(); masks, rejected = temporal_masks(fields, gray, scenes, 10, .5)
            flow_s = time.perf_counter() - t
            results.append({"opencv_threads": threads, "object_detection_and_decode_s": detection_s,
                            "flow_s": flow_s, "masks": sum(map(len, masks)), "rejected": rejected,
                            "object_candidates": objects_count})
            print(results[-1], flush=True)
        del gray
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
