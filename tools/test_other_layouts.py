"""Real OCR smoke test: JSON phrases in two unrelated layouts and colour schemes."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from anonymize import Mask, parser, raster_mask, run


def main():
    out = Path("diagnostics/other_layouts"); out.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path("redaction_data.json").read_text())
    phrases = data["frazy_do_ukrycia"]
    chosen = [phrases[i] for i in (0, 3, 5, 7)]
    font_path = subprocess.check_output(["fc-match", "-f", "%{file}", "sans"], text=True)
    font = ImageFont.truetype(font_path, 28)
    input_path = out / "input.mp4"; output_path = out / "output.mp4"
    writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (800, 450))
    assert writer.isOpened()
    expected = []; foregrounds = []; previews = []
    try:
        for scene in range(2):
            canvas = Image.new("RGB", (800, 450), (25, 34, 54) if scene == 0 else (245, 229, 199))
            draw = ImageDraw.Draw(canvas)
            positions = [(48, 75), (48, 150), (48, 225), (48, 300)] if scene == 0 else [(260, 110), (360, 200), (150, 300), (430, 365)]
            boxes = []
            for text, pos in zip(chosen, positions):
                draw.text(pos, text, font=font, anchor="lt", fill=(245, 245, 245) if scene == 0 else (22, 50, 75))
                boxes.append(draw.textbbox(pos, text, font=font, anchor="lt"))
            frame = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
            background = frame[0, 0].astype(np.int16)
            text_pixels = np.max(np.abs(frame.astype(np.int16) - background), axis=2) > 40
            for _ in range(3):
                writer.write(frame); expected.append(boxes); foregrounds.append(text_pixels)
    finally:
        writer.release()
    args = parser().parse_args(["--input", str(input_path), "--output", str(output_path), "--overwrite"])
    report = run(args)
    assert report["mode"] == "blur" and not report["settings"]["fail_closed"]
    coverages = []; fractions = []
    cap = cv2.VideoCapture(str(output_path))
    try:
        for i, boxes in enumerate(expected):
            row = report["frames"][i]
            masks = [Mask(m["polygon"]) for m in row["masks"]]
            bitmap = raster_mask((450, 800), masks, args.margin)
            fraction = float(np.mean(bitmap > 0)); fractions.append(fraction)
            assert fraction < .18, "Mask expanded beyond the value strips"
            for x1, y1, x2, y2 in boxes:
                text_pixels = foregrounds[i][y1:y2, x1:x2]
                coverage = float(np.mean(bitmap[y1:y2, x1:x2][text_pixels] > 0)); coverages.append(coverage)
                assert coverage > .99, f"Uncovered value in synthetic frame {i}"
            ok, frame = cap.read(); assert ok
            if i in (0, 3):
                previews.append(frame)
    finally:
        cap.release()
    cv2.imwrite(str(out / "preview.jpg"), cv2.vconcat(previews))
    result = {"frames": len(expected), "value_checks": len(coverages), "minimum_coverage": min(coverages),
              "maximum_masked_fraction": max(fractions), "layouts": 2, "mode": report["mode"],
              "scenes_detected": len(set(f["scene"] for f in report["frames"]))}
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
