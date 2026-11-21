"""Offline CPU anonymisation: OCR seeds, checked bidirectional flow, object fallback.

Only geometry, phrase indices and aggregate timings enter diagnostics. Frame numbers
are zero based. No input video is modified; output is committed only after validation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from fractions import Fraction
import importlib.metadata
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata

import cv2
import numpy as np


MONTHS = {name: short for short, names in [
    ("JAN", ("JANUARY",)), ("FEB", ("FEBRUARY",)), ("MAR", ("MARCH",)),
    ("APR", ("APRIL",)), ("MAY", ()), ("JUN", ("JUNE",)),
    ("JUL", ("JULY",)), ("AUG", ("AUGUST",)), ("SEP", ("SEPT", "SEPTEMBER")),
    ("OCT", ("OCTOBER",)), ("NOV", ("NOVEMBER",)), ("DEC", ("DECEMBER",)),
] for name in (short, *names)}
CONFUSIONS = [set("0O"), set("1IL"), set("8B")]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.upper()).translate(str.maketrans({"\u0141": "L", "Ø": "O", "Ð": "D"}))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(MONTHS.get(t, t) for t in re.findall(r"[A-Z0-9]+", text))


def edit_cost(a: str, b: str) -> float:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        new = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else (.2 if any(ca in s and cb in s for s in CONFUSIONS) else 1)
            new.append(min(row[j + 1] + 1, new[-1] + 1, row[j] + cost))
        row = new
    return row[-1]


class Matcher:
    def __init__(self, phrases):
        self.phrases = [normalize(p) for p in phrases]

    def match(self, text):
        tokens = normalize(text).split()
        found = set()
        for idx, phrase in enumerate(self.phrases):
            target = phrase.replace(" ", "")
            for start in range(len(tokens)):
                for end in range(start + 1, min(len(tokens), start + len(phrase.split()) + 3) + 1):
                    candidate = "".join(tokens[start:end])
                    if abs(len(candidate) - len(target)) > 2:
                        continue
                    limit = .41 if len(target) < 6 else min(2, max(1, len(target) * .12))
                    if len(target) < 4:
                        limit = 0
                    if edit_cost(candidate, target) <= limit:
                        found.add(idx)
        return sorted(found)


@dataclass
class Mask:
    poly: np.ndarray
    source: str = "ocr"
    phrase_ids: list = field(default_factory=list)

    def __post_init__(self):
        self.poly = np.asarray(self.poly, np.float32)

    def record(self):
        return {"polygon": np.round(self.poly, 2).tolist(), "source": self.source,
                "phrase_ids": self.phrase_ids}


def hull(points):
    return cv2.convexHull(np.asarray(points, np.float32).reshape(-1, 2)).reshape(-1, 2)


def rect(poly):
    p = np.asarray(poly)
    return np.r_[p.min(axis=0), p.max(axis=0)].astype(float)


def box_poly(box):
    x1, y1, x2, y2 = box
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def iou(a, b):
    left = np.maximum(a[:2], b[:2]); right = np.minimum(a[2:], b[2:])
    intersection = np.maximum(0, right - left).prod()
    return float(intersection / max(1, (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection))


def ocr_masks(results, matcher):
    detections = [(hull(p), str(t)) for p, t, confidence in results if str(t).strip()]
    matches = []
    for p, text in detections:
        ids = matcher.match(text)
        if ids:
            matches.append(Mask(p, "ocr", ids))
    for p, text in detections:
        edges = np.roll(p, -1, axis=0) - p
        u = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        u = u / max(1, np.linalg.norm(u))
        if u[0] < 0:
            u = -u
        v = np.array([-u[1], u[0]])
        pp = p @ u; pv = p @ v
        height = max(2, np.ptp(pv))
        chain = [(p, text)]; right = pp.max()
        for q, other in sorted(detections, key=lambda item: float((item[0] @ u).min())):
            qq = q @ u; qv = q @ v
            gap = qq.min() - right
            if qq.min() <= pp.min() + 1 or gap < -height * .4:
                continue
            if abs(qv.mean() - pv.mean()) > .6 * max(height, np.ptp(qv)):
                continue
            if max(height, np.ptp(qv)) > 2.2 * min(height, max(2, np.ptp(qv))):
                continue
            if gap > height * 2.5 or len(chain) >= 5:
                break
            chain.append((q, other)); right = qq.max()
            ids = matcher.match(" ".join(t for _, t in chain))
            individual_ids = {idx for _, part in chain for idx in matcher.match(part)}
            if ids and not set(ids).issubset(individual_ids):
                matches.append(Mask(hull(np.concatenate([r for r, _ in chain])), "ocr_joined", ids))
    kept = []
    for m in sorted(matches, key=lambda m: cv2.contourArea(m.poly)):
        if any(contained(k.poly, m.poly) and set(m.phrase_ids).issubset(k.phrase_ids) for k in kept):
            continue
        kept.append(m)
    return kept


def scene_changed(a, b):
    small_a = cv2.resize(a, (96, 64)); small_b = cv2.resize(b, (96, 64))
    diff = cv2.absdiff(small_a, small_b)
    if not (diff.mean() > 25 and np.mean(diff > 35) > .55):
        return False
    shift, response = cv2.phaseCorrelate(np.float32(a), np.float32(b))
    return not (response > .4 and math.hypot(*shift) < .2 * math.hypot(*a.shape))


def object_candidates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 10, 30)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = gray.size
    objects = []
    for contour in contours:
        p = hull(contour); area = cv2.contourArea(p)
        b = rect(p); w, h = b[2:] - b[:2]
        if .015 * total < area < .65 * total and min(w, h) >= 16 and area / max(1, w * h) > .3:
            objects.append(p)
    return objects


def checked_flow(a, b, poly, scale):
    p = np.float32(poly) * scale
    support = np.zeros(a.shape, np.uint8)
    cv2.fillConvexPoly(support, np.rint(p).astype(np.int32), 255)
    support = cv2.dilate(support, np.ones((31, 31), np.uint8))
    pts = cv2.goodFeaturesToTrack(a, 90, .01, 3, mask=support)
    if pts is None or len(pts) < 6:
        return None
    params = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, .01))
    next_pts, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, **params)
    if next_pts is None:
        return None
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, next_pts, None, **params)
    if back is None:
        return None
    good = (ok.ravel() == 1) & (back_ok.ravel() == 1) & (np.linalg.norm(pts - back, axis=2).ravel() < 1)
    src, dst = pts[good], next_pts[good]
    if len(src) < 6 or cv2.contourArea(hull(src)) < 12:
        return None
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 1.5)
    if H is None or not np.isfinite(H).all() or inliers.mean() < .7:
        return None
    warped = cv2.perspectiveTransform(p.reshape(1, -1, 2), H).reshape(-1, 2)
    ratio = cv2.contourArea(warped) / max(1, cv2.contourArea(p))
    if not .45 < ratio < 2.2 or not cv2.isContourConvex(warped):
        return None
    if np.max(np.linalg.norm(warped - p, axis=1)) > .15 * math.hypot(*a.shape):
        return None
    projected = cv2.warpPerspective(a, H, (b.shape[1], b.shape[0]))
    check = np.zeros(a.shape, np.uint8)
    cv2.fillConvexPoly(check, np.rint(warped).astype(np.int32), 255)
    if not check.any() or cv2.absdiff(projected, b)[check > 0].mean() > 25:
        return None
    return warped / scale


def surface_quad(poly):
    p = cv2.approxPolyDP(hull(poly), .015 * cv2.arcLength(hull(poly), True), True).reshape(-1, 2)
    if len(p) < 4 or len(p) > 12:
        return None
    q = max((np.float32(p[list(ids)]) for ids in combinations(range(len(p)), 4)), key=cv2.contourArea)
    q = np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0)
    return q


def surface_motion(a, b, source, target, scale):
    source = hull(source) * scale; target = hull(target) * scale
    qa, qb = surface_quad(source), surface_quad(target)
    if qa is None or qb is None or min(cv2.contourArea(qa), cv2.contourArea(qb)) < 100:
        return None
    initial = cv2.getPerspectiveTransform(qa, qb)
    if not np.isfinite(initial).all() or abs(np.linalg.det(initial)) < 1e-10:
        return None
    support = np.zeros(a.shape, np.uint8)
    cv2.fillConvexPoly(support, np.rint(source).astype(np.int32), 255)
    pts = cv2.goodFeaturesToTrack(a, 500, .005, 3, mask=support)
    if pts is None or len(pts) < 12:
        return None
    aligned = cv2.warpPerspective(b, np.linalg.inv(initial), a.shape[::-1])
    guess = pts.copy()
    params = dict(winSize=(21, 21), maxLevel=2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    dst, ok, _ = cv2.calcOpticalFlowPyrLK(a, aligned, pts, guess, **params)
    if dst is None:
        return None
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(aligned, a, dst, pts.copy(), **params)
    if back is None:
        return None
    good = (ok.ravel() > 0) & (back_ok.ravel() > 0) & (np.linalg.norm(pts - back, axis=2).ravel() < 1.2)
    if good.sum() < 12:
        return None
    dst = cv2.perspectiveTransform(dst, initial)
    H, inliers = cv2.findHomography(pts[good], dst[good], cv2.RANSAC, 1.5)
    if H is None or not np.isfinite(H).all() or inliers.sum() < 10 or inliers.mean() < .6:
        return None
    projected = cv2.perspectiveTransform(source[None], H)[0]
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
        return None
    intersection, _ = cv2.intersectConvexConvex(hull(projected), target)
    union = cv2.contourArea(projected) + cv2.contourArea(target) - intersection
    if intersection / max(1, union) < .75:
        return None
    warped = cv2.warpPerspective(a, H, a.shape[::-1])
    check = np.zeros(a.shape, np.uint8)
    cv2.fillConvexPoly(check, np.rint(target).astype(np.int32), 255)
    if cv2.absdiff(warped, b)[check > 0].mean() > 18:
        return None
    return H


def temporal_masks(seeds, grays, scenes, max_gap, scale, candidates=None, diagnostics=None):
    masks = [[Mask(m.poly.copy(), m.source, list(m.phrase_ids)) for m in row] for row in seeds]
    rejected = 0
    surface_cache = {}
    for index, row in enumerate(seeds):
        for seed in row:
            for direction in (-1, 1):
                p = seed.poly.copy(); previous = index
                for distance in range(1, max_gap + 1):
                    j = index + direction * distance
                    if j < 0 or j >= len(seeds) or scenes[j] != scenes[index]:
                        break
                    warped = None
                    if candidates is not None:
                        objects = [(k, obj) for k, obj in enumerate(candidates[previous]) if contained(p, obj)]
                        if objects:
                            k, obj = min(objects, key=lambda item: cv2.contourArea(item[1]))
                            target = max(candidates[j], key=lambda q: iou(rect(obj), rect(q)), default=None)
                            key = (previous, j, k)
                            if target is not None and iou(rect(obj), rect(target)) > .12:
                                if key not in surface_cache:
                                    surface_cache[key] = surface_motion(grays[previous], grays[j], obj, target, scale)
                                H = surface_cache[key]
                                if H is not None:
                                    q = cv2.perspectiveTransform((p * scale)[None], H)[0] / scale
                                    if contained(q, target) and .2 < cv2.contourArea(q) / max(1, cv2.contourArea(p)) < 5:
                                        warped = q
                    if warped is None:
                        warped = checked_flow(grays[previous], grays[j], p, scale)
                    if warped is None:
                        rejected += 1
                        if diagnostics is not None:
                            diagnostics.setdefault("rejected_frames", set()).add(j)
                        break
                    p = warped
                    if any(contained(p, m.poly) for m in seeds[j]):
                        break
                    if not any(all(cv2.pointPolygonTest(m.poly, tuple(map(float, pt)), True) >= -1 for pt in p)
                               for m in masks[j]):
                        masks[j].append(Mask(p.copy(), "flow", seed.phrase_ids))
                    previous = j
    return masks, rejected


def object_masks(candidates, seeds, scenes, enabled=True):
    tracks = []
    for i, polygons in enumerate(candidates):
        available = [t for t in tracks if t["scene"] == scenes[i] and i - t["last"] <= 3]
        used = set()
        for poly in sorted(polygons, key=cv2.contourArea, reverse=True):
            box = rect(poly)
            options = [(iou(box, rect(t["items"][-1][1])), k, t) for k, t in enumerate(available) if k not in used]
            score, k, best = max(options, key=lambda x: x[0], default=(0, -1, None))
            if score < .12:
                best = {"scene": scenes[i], "last": i, "items": [], "evidence": set()}
                tracks.append(best)
            else:
                used.add(k)
            best["last"] = i; best["items"].append((i, poly))
            for mask in seeds[i]:
                if contained(mask.poly, poly):
                    best["evidence"].add(i)
    masks = [[] for _ in seeds]; trusted = []; gaps = []
    if not enabled:
        return masks, trusted, gaps
    for track in tracks:
        boxes = np.array([rect(p) for _, p in track["items"]])
        moving = len(boxes) > 1 and np.max(np.ptp(boxes, axis=0)) > 12
        if len(track["evidence"]) < 2 or not moving:
            continue
        trusted.append({"first_frame": track["items"][0][0], "last_frame": track["last"],
                        "evidence_frames": sorted(track["evidence"])})
        for i, p in track["items"]:
            masks[i].append(Mask(p.copy(), "object_fail_closed"))
        observed = {i for i, _ in track["items"]}
        gaps.extend(i for i in range(min(observed), max(observed) + 1) if i not in observed)
    return masks, trusted, sorted(set(gaps))


def contained(inner, outer):
    outer = np.asarray(outer, np.float32)
    return all(cv2.pointPolygonTest(outer, tuple(map(float, p)), False) >= 0 for p in inner)


def raster_mask(shape, masks, margin):
    canvas = np.zeros(shape[:2], np.uint8)
    for item in masks:
        p = np.asarray(item.poly, np.float32)
        if len(p) >= 3 and np.isfinite(p).all():
            cv2.fillConvexPoly(canvas, np.rint(hull(p)).astype(np.int32), 255)
    if margin:
        canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2))
    return canvas


def apply_masks(frame, masks, mode="blur", margin=5, color=(0, 0, 0), pixel_size=24, blur_sigma=30):
    mask = raster_mask(frame.shape, masks, margin)
    result = frame.copy()
    if mode == "solid":
        result[mask > 0] = color
    else:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for label in range(1, count):
            x, y, w, h, _ = stats[label]; roi = frame[y:y + h, x:x + w]
            if mode == "pixelate":
                small = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_AREA)
                censored = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                censored = cv2.GaussianBlur(roi, (51, 51), blur_sigma)
            region = result[y:y + h, x:x + w]
            inside = labels[y:y + h, x:x + w] == label
            region[inside] = censored[inside]
    return result, mask


def probe(path):
    if not Path(path).is_file():
        raise ValueError("Input file does not exist")
    try:
        proc = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                              check=True, capture_output=True, text=True)
        info = json.loads(proc.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError("FFprobe cannot read the file") from exc
    videos = [s for s in info["streams"] if s["codec_type"] == "video"]
    if not videos:
        raise ValueError("No video stream found")
    video = videos[0]
    try:
        fps = Fraction(video["avg_frame_rate"])
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Invalid frame rate") from exc
    if fps <= 0 or video["width"] < 1 or video["height"] < 1:
        raise ValueError("Invalid video parameters")
    timestamp_probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)],
                                 check=True, capture_output=True, text=True)
    if timestamp_probe.stderr.strip():
        raise ValueError("FFprobe detected a damaged video stream")
    timestamps = timestamp_probe.stdout
    times = [float(row.split(",")[0]) for row in timestamps.splitlines() if row.split(",")[0].strip()]
    if len(times) > 2 and np.max(np.abs(np.diff(times) - 1 / float(fps))) > .002:
        raise ValueError("Variable frame rate video is not supported; a constant frame rate is required")
    return {"width": video["width"], "height": video["height"], "fps": str(fps),
            "frames": len(times), "duration": float(video.get("duration", info.get("format", {}).get("duration", len(times) / float(fps)))),
            "video_codec": video["codec_name"], "pixel_format": video.get("pix_fmt"),
            "audio": [s["codec_name"] for s in info["streams"] if s["codec_type"] == "audio"],
            "start": float(video.get("start_time", 0))}


def load_config(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read the JSON configuration") from exc
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    phrases = data.get("frazy_do_ukrycia")
    if not isinstance(phrases, list) or not phrases or any(not isinstance(p, str) or len(normalize(p).replace(" ", "")) < 3 for p in phrases):
        raise ValueError("frazy_do_ukrycia: a non-empty list of phrases is required (minimum 3 characters)")
    regions = data.get("regions", [])
    if not isinstance(regions, list) or any(not isinstance(r, dict) for r in regions):
        raise ValueError("regions: a list of objects is required")
    for region in regions:
        bounds = region.get("bounds", [])
        if len(bounds) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bounds):
            raise ValueError("regions.bounds: [x,y,width,height] values in the 0–1 range are required")
        x, y, w, h = bounds
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1 or y + h > 1:
            raise ValueError("Region extends beyond the image")
        if region.get("policy", "ocr") not in ("ocr", "always_mask"):
            raise ValueError("regions.policy: expected ocr or always_mask")
        if not isinstance(region.get("scale", 2), (int, float)) or not 1 <= region.get("scale", 2) <= 3:
            raise ValueError("regions.scale: expected a number from 1 to 3")
    return data


class OCR:
    def __init__(self, threads, languages, download=False):
        os.environ["OMP_NUM_THREADS"] = str(threads)
        import torch
        import easyocr
        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        self.reader = easyocr.Reader(languages, gpu=False, verbose=False, download_enabled=download)
        self.detect_s = self.recognize_s = 0
        self.calls = self.boxes = 0

    def read(self, image, scale=1, preprocess="none", tuned=True, rotation=False):
        if scale != 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if preprocess == "clahe":
            gray = cv2.createCLAHE(2, (8, 8)).apply(gray)
        elif preprocess == "threshold":
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
        elif preprocess == "sharpen":
            gray = cv2.addWeighted(gray, 1.6, cv2.GaussianBlur(gray, (0, 0), 1), -.6, 0)
        params = dict(min_size=8, text_threshold=.5, low_text=.25, mag_ratio=1,
                      add_margin=.1, slope_ths=.3, canvas_size=2560) if tuned else {}
        t = time.perf_counter(); horizontal, free = self.reader.detect(gray, **params)
        self.detect_s += time.perf_counter() - t
        t = time.perf_counter()
        results = self.reader.recognize(gray, horizontal[0], free[0], workers=0,
                                       contrast_ths=.2 if tuned else .1, adjust_contrast=.7 if tuned else .5,
                                       rotation_info=[90, 180, 270] if rotation else None)
        self.recognize_s += time.perf_counter() - t
        self.calls += 1; self.boxes += len(results)
        return [(np.asarray(p, np.float32) / scale, text, float(prob)) for p, text, prob in results]


def encode(input_path, destination, meta, masks, args, workspace):
    cap = cv2.VideoCapture(str(input_path))
    video = workspace / "video.mp4"
    pix_fmt = "yuv420p" if meta["width"] % 2 == meta["height"] % 2 == 0 else "yuv444p"
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f'{meta["width"]}x{meta["height"]}', "-r", meta["fps"], "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-threads", str(args.threads), "-preset", "medium", "-crf", str(args.crf),
               "-pix_fmt", pix_fmt, str(video)]
    diagnostic = Path(args.diagnostics) if args.diagnostics else None
    if diagnostic:
        diagnostic.mkdir(parents=True, exist_ok=True)
    with open(workspace / "encoder.log", "w+") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for i, row in enumerate(masks):
                ok, frame = cap.read()
                if not ok or frame.shape[:2] != (meta["height"], meta["width"]):
                    raise ValueError("Decoding failed while writing the output")
                result, bitmap = apply_masks(frame, row, args.mode, args.margin, args.color, args.pixel_size, args.blur_sigma)
                if diagnostic and i in args.diagnostic_frames:
                    preview = result.copy()
                    contours, _ = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(preview, contours, -1, (0, 0, 255), 2)
                    cv2.imwrite(str(diagnostic / f"mask_{i:06d}.png"), preview)
                process.stdin.write(result.tobytes())
            if cap.read()[0]:
                raise ValueError("Inconsistent frame count during the second read")
            process.stdin.close()
            if process.wait() != 0:
                raise ValueError("FFmpeg: H.264 encoding failed")
        finally:
            cap.release()
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
    final = workspace / "final.mp4"
    offset = ["-itsoffset", str(-meta["start"])] if meta["start"] else []
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(video), *offset, "-i", str(input_path),
                    "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-map_metadata", "-1",
                    "-movflags", "+faststart", str(final)], check=True, capture_output=True)
    actual = probe(final)
    if any(actual[k] != meta[k] for k in ("frames", "fps", "width", "height", "audio")):
        raise ValueError("Output validation failed: video or audio parameters do not match")
    if abs(actual["duration"] - meta["frames"] / float(Fraction(meta["fps"]))) > 1 / float(Fraction(meta["fps"])):
        raise ValueError("Output validation failed: video duration does not match")
    os.replace(final, destination)
    return actual


def run(args):
    start = time.perf_counter()
    source = Path(args.input).resolve(); destination = Path(args.output).resolve()
    config_path = Path(args.config).resolve()
    report_path = Path(args.report).resolve() if args.report else destination.with_suffix(".report.json")
    if source == destination:
        raise ValueError("Input and output must be different files")
    if destination == config_path or any(report_path == protected or
            (report_path.exists() and protected.exists() and report_path.samefile(protected))
            for protected in (source, destination, config_path)):
        raise ValueError("The output or report would overwrite the input or configuration file")
    if destination.exists() and not args.overwrite:
        raise ValueError("Output file already exists; choose another name or use --overwrite")
    if not destination.parent.is_dir():
        raise ValueError("Output directory does not exist")
    if not report_path.parent.is_dir():
        raise ValueError("Report directory does not exist")
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            raise ValueError(f"Required executable is missing: {executable}")
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], check=True, capture_output=True, text=True)
    if "libx264" not in encoders.stdout:
        raise ValueError("This FFmpeg installation does not provide the libx264 encoder")
    config = load_config(args.config); matcher = Matcher(config["frazy_do_ukrycia"])
    meta = probe(source)
    if not meta["frames"]:
        raise ValueError("No decodable frames found")
    cv2.setNumThreads(args.opencv_threads)
    t = time.perf_counter(); ocr = OCR(args.threads, args.languages.split(","), args.download_models)
    init_s = time.perf_counter() - t
    print(f'Video: {meta["frames"]} frames, {meta["fps"]} FPS; CPU OCR initialized in {init_s:.2f} s', flush=True)
    with tempfile.TemporaryDirectory(prefix=".anonymize-", dir=destination.parent) as temporary:
        workspace = Path(temporary)
        scale = .5
        small_shape = (round(meta["height"] * scale), round(meta["width"] * scale))
        grays = np.memmap(workspace / "gray.bin", mode="w+", dtype=np.uint8, shape=(meta["frames"], *small_shape))
        seeds = []; candidates = []; scenes = []; scene = 0; object_s = 0
        cap = cv2.VideoCapture(str(source)); t_analysis = time.perf_counter()
        try:
            for i in range(meta["frames"]):
                ok, frame = cap.read()
                if not ok or frame.shape[:2] != (meta["height"], meta["width"]):
                    raise ValueError(f"Damaged video: cannot read frame {i}")
                gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), small_shape[::-1])
                cut = i > 0 and scene_changed(grays[i - 1], gray)
                scene += int(cut); scenes.append(scene); grays[i] = gray
                t = time.perf_counter(); objects = object_candidates(frame); object_s += time.perf_counter() - t
                candidates.append(objects)
                row = []
                if i % args.ocr_every == 0 or cut or i == meta["frames"] - 1:
                    regions = config.get("regions", [])
                    ocr_regions = [r for r in regions if r.get("policy", "ocr") == "ocr"]
                    if not ocr_regions:
                        row = ocr_masks(ocr.read(frame, preprocess=args.preprocess, rotation=args.rotation), matcher)
                    for region in ocr_regions:
                        x, y, w, h = region["bounds"]
                        x1, y1 = int(x * meta["width"]), int(y * meta["height"])
                        x2, y2 = math.ceil((x + w) * meta["width"]), math.ceil((y + h) * meta["height"])
                        results = ocr.read(frame[y1:y2, x1:x2], region.get("scale", args.roi_scale), args.preprocess, rotation=args.rotation)
                        offset = np.float32([x1, y1])
                        row.extend(ocr_masks([(p + offset, text, prob) for p, text, prob in results], matcher))
                    if args.roi_retry:
                        for poly in objects:
                            if any(cv2.pointPolygonTest(poly, tuple(map(float, m.poly.mean(axis=0))), False) >= 0 for m in row):
                                continue
                            x1, y1, x2, y2 = rect(poly).astype(int)
                            results = ocr.read(frame[y1:y2 + 1, x1:x2 + 1], args.roi_scale, args.preprocess, rotation=args.rotation)
                            row.extend(ocr_masks([(p + [x1, y1], text, prob) for p, text, prob in results], matcher))
                seeds.append(row)
                if i % 25 == 0 or i + 1 == meta["frames"]:
                    print(f'\rAnalysis [{i + 1}/{meta["frames"]}] {(i + 1) / meta["frames"]:.0%}', end="", flush=True)
            if cap.read()[0]:
                raise ValueError("FFprobe and the decoder returned different frame counts")
        finally:
            cap.release()
        analysis_s = time.perf_counter() - t_analysis
        print(flush=True)
        t = time.perf_counter()
        objects, trusted, gaps = object_masks(candidates, seeds, scenes, args.fail_closed)
        tracking_diagnostics = {}
        masks, rejected = temporal_masks(seeds, grays, scenes, args.hold_frames, scale, candidates, tracking_diagnostics)
        grays.flush(); del grays
        for i, row in enumerate(masks):
            row.extend(objects[i])
            for region in config.get("regions", []):
                if region.get("policy") == "always_mask":
                    x, y, w, h = region["bounds"]
                    row.append(Mask(box_poly([x * meta["width"], y * meta["height"], (x + w) * meta["width"], (y + h) * meta["height"]]), "configured_region"))
        tracking_s = time.perf_counter() - t
        if gaps:
            raise ValueError("Sensitive object geometry was lost; no output was written. "
                             "After defining an always_mask region manually, use --no-fail-closed")
        print(f'Tracking: {tracking_s:.2f} s; rejected transformations: {rejected}; encoding...', flush=True)
        t = time.perf_counter(); actual = encode(source, destination, meta, masks, args, workspace)
        encoding_s = time.perf_counter() - t
    elapsed = time.perf_counter() - start
    report = {"input": meta, "output": actual, "mode": args.mode, "threads": args.threads,
              "settings": {"margin": args.margin, "hold_frames": args.hold_frames, "roi_scale": args.roi_scale,
                           "roi_retry": args.roi_retry, "preprocess": args.preprocess, "rotation": args.rotation,
                           "fail_closed": args.fail_closed, "crf": args.crf},
              "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "opencv": cv2.__version__,
                          "torch": importlib.metadata.version("torch"), "torchvision": importlib.metadata.version("torchvision"),
                          "easyocr": importlib.metadata.version("easyocr")},
              "opencv_threads": args.opencv_threads, "ocr_every": args.ocr_every,
              "frames_analyzed": len(seeds), "frames_with_ocr_masks": sum(bool(r) for r in seeds),
              "frames_with_masks": sum(bool(r) for r in masks), "detected_regions": sum(map(len, seeds)),
              "masked_regions": sum(map(len, masks)),
              "uncertainty_frames": len(tracking_diagnostics.get("rejected_frames", set())),
              "tracking_rejection_frames": sorted(tracking_diagnostics.get("rejected_frames", set())),
              "rejected_flow_chains": rejected, "object_geometry_gaps": gaps, "trusted_objects": trusted,
              "timing_s": {"initialization": init_s, "analysis": analysis_s, "ocr_detector": ocr.detect_s,
                           "ocr_recognizer": ocr.recognize_s, "object_detection": object_s,
                           "tracking": tracking_s, "encoding": encoding_s, "total": elapsed,
                           "per_frame": elapsed / len(seeds)},
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "ocr_calls": ocr.calls, "ocr_boxes": ocr.boxes,
              "frames": [{"frame": i, "scene": scenes[i], "masks": [m.record() for m in row]} for i, row in enumerate(masks)]}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f'Wrote {destination.name}: {len(seeds)} frames, {report["masked_regions"]} masks; '
          f'{elapsed:.2f} s ({elapsed / len(seeds):.3f} s/frame); audio: {actual["audio"]}', flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="recording.mp4")
    p.add_argument("--output", default="recording_redacted_v2.mp4")
    p.add_argument("--config", default="redaction_data.json")
    p.add_argument("--mode", choices=("solid", "pixelate", "blur"), default="blur")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--opencv-threads", type=int, default=1)
    p.add_argument("--ocr-every", type=int, default=5)
    p.add_argument("--hold-frames", type=int, default=10)
    p.add_argument("--margin", type=int, default=5)
    p.add_argument("--roi-scale", type=float, default=2)
    p.add_argument("--roi-retry", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fail-closed", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--preprocess", choices=("none", "clahe", "threshold", "sharpen"), default="none")
    p.add_argument("--rotation", action="store_true")
    p.add_argument("--languages", default="en")
    p.add_argument("--download-models", action="store_true")
    p.add_argument("--color", type=lambda s: tuple(int(c) for c in s.split(",")), default=(0, 0, 0), help="B,G,R")
    p.add_argument("--pixel-size", type=int, default=24)
    p.add_argument("--blur-sigma", type=float, default=30)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--report")
    p.add_argument("--diagnostics")
    p.add_argument("--diagnostic-frames", type=lambda s: [int(x) for x in s.split(",")],
                   default=[40, 41, 42, 43, 50, 52, 53, 54, 141, 142, 143, 164, 165, 166])
    p.add_argument("--overwrite", action="store_true")
    return p


def main():
    p = parser(); args = p.parse_args()
    if min(args.threads, args.opencv_threads, args.ocr_every, args.pixel_size) < 1 or args.hold_frames < args.ocr_every:
        p.error("Thread counts and intervals must be positive; hold-frames >= ocr-every")
    if args.margin < 0 or not 1 <= args.roi_scale <= 3 or not 0 <= args.crf <= 51 or args.blur_sigma <= 0:
        p.error("Invalid mask, OCR, or encoding parameters")
    if len(args.color) != 3 or any(not 0 <= c <= 255 for c in args.color):
        p.error("Color must contain three numbers in the 0–255 range")
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(f"Error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
