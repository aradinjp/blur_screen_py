import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from anonymize import (Mask, Matcher, apply_masks, box_poly, checked_flow, load_config,
                     normalize, object_masks, ocr_masks, parser, probe, run,
                     scene_changed, temporal_masks, surface_motion)


def test_normalize_unicode_spacing_months():
    assert normalize("  \u017c\u00d3\u0141\u0106—12 JULY / 1998 ") == "ZOLC 12 JUL 1998"
    assert normalize("Zoé 12 JULY/1998") == "ZOE 12 JUL 1998"
    assert normalize("  12\tJul.\n1998 ") == "12 JUL 1998"


def test_match_confusions_and_boundaries():
    matcher = Matcher(["BOD12873", "ALICE", "SAMPLECODE", "12 July 1998"])
    assert 0 in matcher.match("8ODI2873")
    assert 1 in matcher.match("AL1CE")
    assert 2 in matcher.match("SAMPLEC0DE")
    assert 3 in matcher.match("12 JUL I998")
    assert not matcher.match("MALICE")
    assert not matcher.match("BOB")
    assert not matcher.match("13 JUL 2099")
    assert not matcher.match("country POL")


def result(text, x, y, w=20, h=12):
    return (box_poly([x, y, x + w, y + h]), text, .4)


def test_fragmented_date_joins_only_adjacent_row():
    matcher = Matcher(["12 July 1998"])
    masks = ocr_masks([result("12", 10, 10), result("Jul", 34, 10), result("1998", 58, 10, 34)], matcher)
    assert len(masks) == 1 and masks[0].source == "ocr_joined"
    np.testing.assert_array_equal(masks[0].poly.min(0), [10, 10])
    np.testing.assert_array_equal(masks[0].poly.max(0), [92, 22])
    assert not ocr_masks([result("12", 10, 10), result("Jul", 34, 50), result("1998", 58, 90)], matcher)
    assert not ocr_masks([result("12", 10, 10), result("Jul", 340, 10), result("1998", 580, 10)], matcher)


@pytest.mark.parametrize("mode", ["solid", "pixelate", "blur"])
def test_clipping_all_vertices_and_margin(mode):
    image = np.random.default_rng(2).integers(30, 255, (50, 60, 3), dtype=np.uint8)
    polygon = np.float32([[-20, 3], [17, -9], [40, 20], [0, 38]])
    output, bitmap = apply_masks(image, [Mask(polygon)], mode, 3)
    assert bitmap[1, 1] and bitmap[19, 38]
    assert not bitmap[49, 59]
    np.testing.assert_array_equal(output[bitmap == 0], image[bitmap == 0])
    if mode == "solid":
        assert not output[bitmap > 0].any()


def moving_frames():
    random = np.random.default_rng(42).integers(0, 256, (120, 160), dtype=np.uint8)
    return [cv2.warpAffine(random, np.float32([[1, 0, i * 2], [0, 1, i]]), (160, 120)) for i in range(5)]


def test_flow_recovers_translation_and_dropped_detections_both_ways():
    frames = moving_frames(); polygon = box_poly([40, 40, 85, 70])
    transformed = checked_flow(frames[0], frames[1], polygon, 1)
    assert transformed is not None
    np.testing.assert_allclose(transformed, polygon + [2, 1], atol=.6)
    seeds = [[Mask(polygon)], [], [], [], [Mask(polygon + [8, 4])]]
    masks, _ = temporal_masks(seeds, frames, [0] * 5, 4, 1)
    assert all(masks)
    for i, row in enumerate(masks):
        assert any(np.max(abs(m.poly.mean(0) - (polygon.mean(0) + [2 * i, i]))) < 1 for m in row)
    reverse, _ = temporal_masks([[], [], [], [], seeds[-1]], frames, [0] * 5, 4, 1)
    assert all(reverse)


def test_scene_cut_blocks_masks_and_unreliable_geometry():
    frames = moving_frames(); p = box_poly([40, 40, 85, 70])
    masks, _ = temporal_masks([[Mask(p)], [], [], [], []], frames, [0, 0, 1, 1, 1], 4, 1)
    assert masks[1] and not any(masks[2:])
    assert scene_changed(np.zeros((100, 100), np.uint8), np.full((100, 100), 255, np.uint8))
    assert not scene_changed(frames[0], frames[1])
    assert checked_flow(frames[0], np.zeros_like(frames[0]), p, 1) is None


def test_moving_object_trust_never_crosses_scene_or_masks_static_panel():
    objects = [[box_poly([10 + 5 * i, 30, 90 + 5 * i, 100])] for i in range(5)]
    seeds = [[Mask(box_poly([25, 40, 45, 50]))], [], [], [Mask(box_poly([40, 40, 60, 50]))], []]
    masks, trusted, gaps = object_masks(objects, seeds, [0] * 5)
    assert all(masks) and len(trusted) == 1 and not gaps
    masks, _, _ = object_masks(objects, seeds, [0, 0, 1, 1, 1])
    assert not any(masks)
    static = [[objects[0][0]]] * 5
    masks, _, _ = object_masks(static, seeds, [0] * 5)
    assert not any(masks)
    masks, _, _ = object_masks(objects, seeds, [0] * 5, enabled=False)
    assert not any(masks)


def test_invalid_input_and_config(tmp_path):
    with pytest.raises(ValueError, match="Input file does not exist"):
        probe(tmp_path / "missing.mp4")
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"not a movie")
    with pytest.raises(ValueError, match="FFprobe"):
        probe(bad)
    cfg = tmp_path / "cfg.json"
    for data in ([], {}, {"frazy_do_ukrycia": [""]}, {"frazy_do_ukrycia": [123]},
                 {"frazy_do_ukrycia": ["ALICE"], "regions": "bad"},
                 {"frazy_do_ukrycia": ["ALICE"], "regions": [{"bounds": [0, 0, .5, .5], "scale": "bad"}]},
                 {"frazy_do_ukrycia": ["ALICE"], "regions": [{"bounds": [0, 0, 2, 1]}]}):
        cfg.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            load_config(cfg)


@pytest.mark.parametrize("audio", [False, True])
def test_video_roundtrip_and_audio_without_real_ocr(tmp_path, monkeypatch, audio):
    import anonymize
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg not installed")
    source = tmp_path / "in.mp4"
    command = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30000/1001:duration=0.3"]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=330:duration=0.3", "-c:a", "aac"]
    subprocess.run(command + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True)

    class FakeOCR:
        detect_s = recognize_s = calls = boxes = 0
        def __init__(self, *args):
            pass
        def read(self, *args, **kwargs):
            return [result("ALICE", 40, 40, 50, 18)]

    monkeypatch.setattr(anonymize, "OCR", FakeOCR)
    cfg = tmp_path / "config.json"; cfg.write_text(json.dumps({"frazy_do_ukrycia": ["ALICE"]}))
    output = tmp_path / "out.mp4"
    args = parser().parse_args(["--input", str(source), "--output", str(output), "--config", str(cfg), "--ocr-every", "2", "--mode", "solid"])
    report = run(args)
    original = probe(source); actual = probe(output)
    assert all(original[k] == actual[k] for k in ("width", "height", "frames", "fps", "audio"))
    assert actual["audio"] == (["aac"] if audio else [])
    if audio:
        packets = lambda path: subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                                               "-c:a", "copy", "-f", "data", "-"], check=True, capture_output=True).stdout
        assert packets(source) == packets(output)
    assert report["frames_with_masks"] == actual["frames"]
    cap = cv2.VideoCapture(str(output)); ok, frame = cap.read(); cap.release()
    assert ok and frame[44:54, 44:84].max() < 10
    assert not list(tmp_path.glob(".anonymize-*"))


def test_failed_encode_leaves_existing_destination_intact(tmp_path):
    output = tmp_path / "out.mp4"; output.write_bytes(b"previous output")
    args = parser().parse_args(["--output", str(output)])
    with pytest.raises(ValueError, match="already exists"):
        run(args)
    assert output.read_bytes() == b"previous output"


def test_perspective_flow_uses_all_corners():
    image = moving_frames()[0]
    polygon = box_poly([40, 35, 95, 80])
    H = np.float32([[1.02, .02, 1.3], [-.015, 1.01, 1], [.00015, -.0001, 1]])
    target = cv2.warpPerspective(image, H, (160, 120))
    actual = checked_flow(image, target, polygon, 1)
    expected = cv2.perspectiveTransform(polygon[None], H)[0]
    assert actual is not None
    np.testing.assert_allclose(actual, expected, atol=1)


def test_object_geometry_gap_is_reported_not_interpolated():
    candidates = [[box_poly([10, 30, 90, 100])], [], [box_poly([25, 30, 105, 100])]]
    seeds = [[Mask(box_poly([30, 40, 50, 50]))], [], [Mask(box_poly([45, 40, 65, 50]))]]
    masks, trusted, gaps = object_masks(candidates, seeds, [0] * 3)
    assert trusted and gaps == [1] and not masks[1]


@pytest.mark.parametrize("option", ["--output", "--report"])
def test_source_cannot_be_overwritten_even_by_report(tmp_path, option):
    source = tmp_path / "original.mp4"; source.write_bytes(b"original data")
    args = parser().parse_args(["--input", str(source), option, str(source), "--overwrite"])
    with pytest.raises(ValueError):
        run(args)
    assert source.read_bytes() == b"original data"


def test_exception_releases_capture_and_removes_disk_cache(tmp_path, monkeypatch):
    import anonymize
    source = tmp_path / "original.mp4"; source.touch()
    cfg = tmp_path / "config.json"; cfg.write_text(json.dumps({"frazy_do_ukrycia": ["ALICE"]}))
    released = []

    class Capture:
        def __init__(self, *args):
            pass
        def read(self):
            return True, np.zeros((100, 100, 3), np.uint8)
        def release(self):
            released.append(True)

    class FailingOCR:
        def __init__(self, *args):
            pass
        def read(self, *args, **kwargs):
            raise RuntimeError("injected OCR failure")

    monkeypatch.setattr(anonymize.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(anonymize, "OCR", FailingOCR)
    monkeypatch.setattr(anonymize, "probe", lambda path: {"frames": 2, "width": 100, "height": 100, "fps": "30"})
    args = parser().parse_args(["--input", str(source), "--output", str(tmp_path / "out.mp4"), "--config", str(cfg)])
    with pytest.raises(RuntimeError, match="injected"):
        run(args)
    assert released and not list(tmp_path.glob(".anonymize-*")) and not (tmp_path / "out.mp4").exists()


def test_report_hardlink_cannot_destroy_original(tmp_path):
    source = tmp_path / "original.mp4"; source.write_bytes(b"original data")
    alias = tmp_path / "report.json"; alias.hardlink_to(source)
    args = parser().parse_args(["--input", str(source), "--report", str(alias), "--output", str(tmp_path / "out.mp4")])
    with pytest.raises(ValueError):
        run(args)
    assert source.read_bytes() == b"original data"


def test_vfr_rejected_instead_of_silently_retimed(tmp_path):
    video = tmp_path / "vfr.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=30:duration=0.2",
                    "-vf", "setpts='if(lt(N,3),N,N+1)/(30*TB)'", "-fps_mode", "vfr", "-c:v", "libx264", str(video)], check=True)
    with pytest.raises(ValueError, match="Variable frame rate"):
        probe(video)


def test_default_preserves_blurred_strips_and_does_not_hide_card():
    args = parser().parse_args([])
    assert args.mode == "blur" and args.margin == 5 and not args.fail_closed
    matcher = Matcher(["EXAMPLE"])
    masks = ocr_masks([result("EXAMPLE", 10, 10, 65, 15), result("description", 90, 10, 90, 15)], matcher)
    assert len(masks) == 1
    assert masks[0].poly[:, 0].max() == 75


def test_surface_motion_transports_only_a_strip_and_validates_perspective():
    image = np.full((240, 320), 215, np.uint8)
    quad = box_poly([40, 35, 280, 205])
    image[35:206, 40:281] = np.random.default_rng(3).integers(30, 200, (171, 241), dtype=np.uint8)
    target_quad = np.float32([[105, 20], [225, 50], [225, 190], [105, 220]])
    H = cv2.getPerspectiveTransform(quad, target_quad)
    target = cv2.warpPerspective(image, H, (320, 240), borderValue=215)
    motion = surface_motion(image, target, quad, target_quad, 1)
    assert motion is not None
    strip = box_poly([130, 105, 210, 118])
    projected = cv2.perspectiveTransform(strip[None], motion)[0]
    expected = cv2.perspectiveTransform(strip[None], H)[0]
    np.testing.assert_allclose(projected, expected, atol=1.5)
    assert cv2.contourArea(projected) < .06 * cv2.contourArea(target_quad)
    assert surface_motion(image, np.full_like(image, 215), quad, target_quad, 1) is None
