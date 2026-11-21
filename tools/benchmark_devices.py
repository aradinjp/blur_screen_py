import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from anonymize import Mask, OCR, load_config, ocr_masks, Matcher, parser, raster_mask, run
from hardware import hardware_info, peak_rss_mb


def audio_hash(path):
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                                "-c:a", "copy", "-f", "data", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    digest = hashlib.sha256()
    try:
        for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
            digest.update(chunk)
        if process.wait() != 0:
            raise RuntimeError("Audio packet verification failed")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    return digest.hexdigest()


def worker(args):
    info = hardware_info(args.device)
    cv2.setNumThreads(1)
    config = load_config(args.config)
    matcher = Matcher(config["frazy_do_ukrycia"])
    start = time.perf_counter()
    ocr = OCR(args.threads, ["en"], device=args.device)
    initialization = time.perf_counter() - start
    cap = cv2.VideoCapture(args.input)
    samples = []
    try:
        ocr.smoke_test()
        for frame_id in args.frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Cannot read a benchmark sample frame")
            start = time.perf_counter()
            results = ocr.read(frame)
            ocr.synchronize()
            samples.append({"frame": frame_id, "inference_s": time.perf_counter() - start,
                            "masks": [m.record() for m in ocr_masks(results, matcher)]})
        info.update(selected_device=ocr.device, mps_pipeline_tested=ocr.mps_tested,
                    ocr_pipeline_tested=True)
    finally:
        cap.release()
        ocr.close()
    result = {"hardware": info, "initialization_s": initialization, "warmup_iterations": 1,
              "samples": samples, "mean_inference_s": float(np.mean([s["inference_s"] for s in samples])),
              "peak_process_rss_mb": peak_rss_mb()}
    if not args.samples_only:
        options = parser().parse_args(["--input", args.input, "--config", args.config,
            "--output", str(Path(args.output).with_suffix(".mp4")), "--device", args.device,
            "--threads", str(args.threads)])
        report = run(options)
        result.update(video_report=str(Path(options.output).with_suffix(".report.json")),
                      total_video_s=report["timing_s"]["total"],
                      effective_fps=report["frames_analyzed"] / report["timing_s"]["total"],
                      peak_process_rss_mb=peak_rss_mb(),
                      audio_packets_preserved=audio_hash(args.input) == audio_hash(options.output)
                      if report["input"]["audio"] else True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


def compare_reports(cpu_path, mps_path):
    cpu = json.loads(Path(cpu_path).read_text())
    mps = json.loads(Path(mps_path).read_text())
    if any(cpu["output"][key] != mps["output"][key] for key in ("width", "height", "frames", "fps", "audio")):
        raise RuntimeError("CPU and MPS output metadata differ")
    rows = []
    for a, b in zip(cpu["frames"], mps["frames"]):
        shape = (cpu["output"]["height"], cpu["output"]["width"])
        masks = []
        for report, row in ((cpu, a), (mps, b)):
            masks.append(raster_mask(shape, [Mask(m["polygon"]) for m in row["masks"]], report["settings"]["margin"]) > 0)
        missed = int(np.count_nonzero(masks[0] & ~masks[1]))
        extra = int(np.count_nonzero(masks[1] & ~masks[0]))
        if missed or extra:
            rows.append({"frame": a["frame"], "cpu_pixels_missing_on_mps": missed,
                         "additional_mps_pixels": extra})
    return {"different_mask_frames": rows, "human_review_required": True,
            "note": "CPU masks are a comparison baseline, not proof of complete anonymization"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="recording.mp4")
    p.add_argument("--config", default="redaction_data.json")
    p.add_argument("--devices", default="cpu,mps")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--frames", type=lambda s: [int(x) for x in s.split(",")], default=[0, 40, 50, 141, 164])
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--output-dir", default="diagnostics/device_benchmark")
    p.add_argument("--samples-only", action="store_true")
    p.add_argument("--device", choices=("cpu", "mps", "auto"))
    p.add_argument("--output")
    args = p.parse_args()
    if args.repeats < 1 or not args.frames:
        p.error("At least one repeat and sample frame are required")
    if args.device:
        if not args.output:
            p.error("Worker requires --output")
        worker(args)
        return
    devices = args.devices.split(",")
    if any(d not in ("cpu", "mps", "auto") for d in devices):
        p.error("Devices must be cpu, mps, or auto")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for iteration in range(args.repeats):
        reports = {}
        for device in devices:
            destination = out / f"{device}_{iteration}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--device", device,
                       "--input", args.input, "--config", args.config, "--threads", str(args.threads),
                       "--frames", ",".join(map(str, args.frames)), "--output", str(destination)]
            if args.samples_only:
                command.append("--samples-only")
            subprocess.run(command, check=True)
            result = json.loads(destination.read_text())
            results.append({"iteration": iteration, "device": device, **result})
            if "video_report" in result:
                reports[device] = result["video_report"]
        if "cpu" in reports and "mps" in reports:
            comparison = compare_reports(reports["cpu"], reports["mps"])
            (out / f"mask_comparison_{iteration}.json").write_text(json.dumps(comparison, indent=2))
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"Benchmark results: {out}")


if __name__ == "__main__":
    main()
