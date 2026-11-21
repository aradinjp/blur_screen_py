"""Separate detector/recognizer/matcher ablations; records no recognized strings."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from anonymize import Matcher, OCR, ocr_masks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="recording.mp4")
    p.add_argument("--config", default="redaction_data.json")
    p.add_argument("--output", default="diagnostics/ocr_stages.json")
    args = p.parse_args()
    cv2.setNumThreads(1); ocr = OCR(6, ["en"])
    phrases = json.loads(Path(args.config).read_text())["frazy_do_ukrycia"]
    matcher = Matcher(phrases); cap = cv2.VideoCapture(args.input); rows = []

    def record(results):
        return [{"polygon": np.asarray(poly).tolist(), "confidence": float(confidence),
                 "exact_phrase_ids": [i for i, p in enumerate(phrases) if p.upper() in text.upper()],
                 "normalized_phrase_ids": matcher.match(text)} for poly, text, confidence in results]

    try:
        for idx in [40, 50, 141, 164]:
            cap.set(1, idx); ok, frame = cap.read(); assert ok
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, f = ocr.reader.detect(frame)
            for name, horizontal, free, contrast in [
                    ("default", h[0], f[0], .1), ("contrast_only", h[0], f[0], .2)]:
                results = ocr.reader.recognize(gray, horizontal, free, contrast_ths=contrast,
                                               adjust_contrast=.5 if name == "default" else .7)
                rows.append({"frame": idx, "variant": name, "boxes": record(results),
                             "matches": [m.record() for m in ocr_masks(results, matcher)]})
            results = ocr.read(frame)
            rows.append({"frame": idx, "variant": "tuned_detector_and_recognizer",
                         "boxes": record(results), "matches": [m.record() for m in ocr_masks(results, matcher)]})
        cap.set(1, 40); _, frame = cap.read()
        results = ocr.reader.recognize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), [[575, 660, 721, 751]], [])
        rows.append({"frame": 40, "variant": "manual_date_box", "boxes": record(results),
                     "matches": [m.record() for m in ocr_masks(results, matcher)]})
    finally:
        cap.release()
    Path(args.output).write_text(json.dumps(rows, indent=2))
    print("Saved stage diagnostics; recognized strings omitted")


if __name__ == "__main__":
    main()
