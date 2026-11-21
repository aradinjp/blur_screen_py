"""Pixel/audio regression check with independent, manually reviewed sample areas.

This is not a certificate of complete anonymisation. Contact sheets require human
review at original resolution before publication. Comparison images contain PII.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from anonymize import Mask, probe, raster_mask


def audio_hash(path):
    data = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c:a", "copy", "-f", "data", "-"],
                          check=True, capture_output=True).stdout
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="recording.mp4")
    p.add_argument("--old", default="recording_redacted_legacy.mp4")
    p.add_argument("--new", default="recording_redacted_v2.mp4")
    p.add_argument("--report", default="recording_redacted_v2.report.json")
    p.add_argument("--baseline", default="diagnostics/baseline_full.json")
    p.add_argument("--annotations", default="tests/recording_annotations.json")
    p.add_argument("--output-dir", default="diagnostics/verification")
    args = p.parse_args()
    cv2.setNumThreads(1)
    report = json.loads(Path(args.report).read_text()); annotations = json.loads(Path(args.annotations).read_text())
    original = probe(args.input); actual = probe(args.new)
    assert all(original[k] == actual[k] for k in ("width", "height", "frames", "fps", "audio"))
    assert abs(original["duration"] - actual["duration"]) < .001
    audio = {"original": audio_hash(args.input), "new": audio_hash(args.new)} if original["audio"] else {}
    assert not audio or audio["original"] == audio["new"]
    baseline = json.loads(Path(args.baseline).read_text()) if Path(args.baseline).is_file() else None
    old_boxes = {f["frame"]: f["exact_masks"] for f in baseline["frames"]} if baseline else {}
    caps = [cv2.VideoCapture(path) for path in (args.input, args.old, args.new)]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    selected = {int(k) for k in annotations["card_value_rectangles"]} | set(annotations["edge_on_frames_without_readable_values"])
    errors = []; regression_errors = []; padding_differences = []; values = []; tiles = []
    squared_error = 0.; unchanged_pixels = 0
    try:
        for i in range(original["frames"]):
            frames = []
            for cap in caps:
                ok, f = cap.read(); assert ok, f"Decode failed at frame {i}"
                frames.append(f)
            new = frames[-1]
            row = report["frames"][i]
            masks = [Mask(np.float32(m["polygon"])) for m in row["masks"]]
            bitmap = raster_mask(new.shape, masks, report.get("settings", {}).get("margin", 10))
            unchanged = cv2.dilate(bitmap, np.ones((9, 9), np.uint8)) == 0
            delta = np.float32(frames[0][unchanged]) - np.float32(new[unchanged])
            squared_error += float(np.square(delta).sum()); unchanged_pixels += delta.size
            areas = list(annotations["static_value_rectangles"])
            if str(i) in annotations["card_value_rectangles"]:
                areas.extend(annotations["card_value_rectangles"][str(i)])
            for area_id, (x1, y1, x2, y2) in enumerate(areas):
                pixels = new[y1:y2, x1:x2]
                covered = float(np.mean(bitmap[y1:y2, x1:x2] > 0))
                source_gray = cv2.cvtColor(frames[0][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                output_gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
                detail_ratio = float(cv2.Laplacian(output_gray, cv2.CV_32F).var() /
                                     max(1, cv2.Laplacian(source_gray, cv2.CV_32F).var()))
                values.append({"frame": i, "area": area_id, "coverage": covered, "remaining_detail_ratio": detail_ratio})
                if covered < .99 or (report["mode"] == "blur" and detail_ratio > .4):
                    errors.append(values[-1])
            for old in old_boxes.get(i, []):
                control = raster_mask(new.shape, [Mask(np.float32(old["polygon"]))], 0)
                coverage = float(np.mean(bitmap[control > 0] > 0))
                if coverage < .995:
                    core = cv2.erode(control, np.ones((5, 5), np.uint8))
                    core_coverage = float(np.mean(bitmap[core > 0] > 0)) if core.any() else coverage
                    item = {"frame": i, "coverage": coverage, "core_coverage": core_coverage, "phrase_ids": old["phrase_ids"]}
                    padding_differences.append(item)
                    if core_coverage < .995:
                        regression_errors.append(item)
            tile = cv2.resize(new, (258, 276)); cv2.putText(tile, str(i), (5, 20), 0, .6, (0, 0, 255), 2)
            tiles.append(tile)
            if len(tiles) == 30 or i == original["frames"] - 1:
                tiles.extend([np.full_like(tile, 255)] * (30 - len(tiles)))
                cv2.imwrite(str(out / f"atlas_{i // 30:02d}.jpg"), cv2.vconcat([cv2.hconcat(tiles[j:j + 5]) for j in range(0, 30, 5)]))
                tiles = []
            if i in selected:
                pieces = []
                for label, image in zip(("original", "previous", "new"), frames):
                    crop = image[475:1025, 170:870].copy()
                    cv2.putText(crop, f"{i} {label}", (6, 24), 0, .7, (0, 0, 255), 2)
                    pieces.append(crop)
                cv2.imwrite(str(out / f"comparison_{i:03d}.png"), cv2.hconcat(pieces))
    finally:
        for cap in caps:
            cap.release()
    result = {"frames_checked": original["frames"], "manual_sample_frames": sorted(selected),
              "static_area_checks": original["frames"] * len(annotations["static_value_rectangles"]),
              "audio_packet_sha256": audio, "manual_area_failures": errors,
              "previous_ocr_box_regression_failures": regression_errors,
              "previous_ocr_box_padding_differences": padding_differences,
              "baseline_regression_performed": baseline is not None, "pixel_checks": values,
              "unmasked_psnr_db": float(10 * np.log10(255 ** 2 / max(1e-12, squared_error / max(1, unchanged_pixels)))),
              "limitation": "Manual rectangles cover selected card frames and static values. Atlas review and human full-resolution review remain necessary."}
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k not in ("pixel_checks", "limitation", "manual_area_failures", "previous_ocr_box_regression_failures")})
    print("Manual area failures:", len(errors), "previous box differences:", len(regression_errors))
    return int(bool(errors or regression_errors))


if __name__ == "__main__":
    raise SystemExit(main())
