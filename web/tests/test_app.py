"""ทดสอบเว็บสาธิตด้วยโมเดลจำลอง (ไม่ต้องมี best.pt หรือ ultralytics)

    python -m pytest web/tests -q
"""

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web"))
import app as web  # noqa: E402

RUN = "B0_s42"


class FakeBoxes:
    def __init__(self, rows):
        a = np.asarray(rows, dtype=float).reshape(-1, 6)
        self.xyxy, self.cls, self.conf = a[:, :4], a[:, 4], a[:, 5]


class FakeResult:
    path = "image0.jpg"
    speed = {"preprocess": 1.0, "inference": 10.0, "postprocess": 0.5}

    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class FakeModel:
    """คืนกรอบตายตัว 3 กรอบ: เหนือ threshold, เท่ากับ threshold พอดี, ต่ำกว่า"""

    def __init__(self, thr):
        self.thr, self.calls = thr, []

    def predict(self, img, **kw):
        self.calls.append(kw)
        return [FakeResult([[10, 10, 50, 50, 1, 0.9], [60, 60, 90, 90, 4, self.thr], [5, 5, 9, 9, 2, self.thr - 0.01]])]


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    info = web.load_model_info({"run_id": RUN})
    w = tmp_path / "best.pt"
    w.write_bytes(b"not the real weights")
    cfg = {"run_id": RUN, "weights": str(w), "temporary": True}
    model = FakeModel(info["threshold"])
    return cfg, model, info


def client(cfg, model, dev):
    return TestClient(web.create_app(cfg, model=model, dev=dev))


def test_info_from_registry_and_thresholds():
    info = web.load_model_info({"run_id": RUN})
    assert info["policy"] == "B0" and info["train_seed"] == 42 and info["imgsz"] == 640
    assert info["threshold"] == 0.5 and info["expected_sha256"].startswith("2864626f")


def test_class_info_english_and_thai_from_mapping():
    ci = web.load_class_info()
    assert [c["code"] for c in ci] == web.CLASS_NAMES
    assert ci[0]["en"] == "pitted surface" and ci[0]["th"] == "ผิวเป็นหลุม"
    assert ci[4]["en"] == "crazing" and ci[4]["th"] == "รอยแตกลายงา" and all(c["th"] and c["desc"] for c in ci)


def test_wrong_sha_refuses_to_start(setup):
    cfg, model, _ = setup
    with pytest.raises(SystemExit):
        web.create_app(cfg, model=model, dev=False)


def test_dev_mode_flags_unverified(setup):
    cfg, model, _ = setup
    r = client(cfg, model, dev=True).get("/model_info").json()
    assert r["verified"] is False and r["temporary"] is True and "expected_sha256" not in r


def test_predict_uses_protocol_settings_and_val_threshold(setup):
    cfg, model, info = setup
    gray = np.full((200, 200, 3), 120, np.uint8)
    r = client(cfg, model, dev=True).post("/predict", files={"file": ("a.png", png(gray))}).json()
    assert model.calls[-1] == {"imgsz": 640, "device": "cpu", "verbose": False, "conf": 0.001, "iou": 0.7, "max_det": 300}
    assert r["count"] == 2  # >= threshold เหมือน evaluator; ต่ำกว่าถูกตัด
    assert [d["class_name"] for d in r["detections"]] == ["class_B", "class_E"]
    assert r["model_ms"] == 11.5 and r["warnings"] == []


def test_domain_warnings_color_and_size(setup):
    cfg, model, _ = setup
    img = np.zeros((900, 1600, 3), np.uint8)
    img[..., 2] = 200
    r = client(cfg, model, dev=True).post("/predict", files={"file": ("c.png", png(img))}).json()
    assert len(r["warnings"]) == 2


@pytest.mark.parametrize("data", [b"", b"%PDF-1.4 not an image", b"\x89PNG\r\n\x1a\n broken"])
def test_bad_files_give_readable_error(setup, data):
    cfg, model, _ = setup
    r = client(cfg, model, dev=True).post("/predict", files={"file": ("x.pdf", data)})
    assert r.status_code == 400 and "อ่านไฟล์นี้เป็นภาพไม่ได้" in r.json()["detail"]


def test_upload_too_large(setup):
    cfg, model, _ = setup
    cfg["max_upload_mb"] = 1
    r = client(cfg, model, dev=True).post("/predict", files={"file": ("big.png", b"0" * (1024 * 1024 + 10))})
    assert r.status_code == 413


def test_model_choice_uses_that_models_threshold_and_unknown_is_404(setup, tmp_path):
    cfg, model, _ = setup
    other = "V8_B0_s42"
    root = tmp_path / "repo"
    (root / "web" / "weights" / other).mkdir(parents=True)
    (root / "web" / "weights" / other / "best.pt").write_bytes(b"other weights")
    (root / "research" / "reports").mkdir(parents=True)
    for f in ("registry.csv", "reports/thresholds_val.json"):
        (root / "research" / f).write_bytes((ROOT / "research" / f).read_bytes())
    (root / "web" / "weights" / RUN).mkdir()
    (root / "web" / "weights" / RUN / "best.pt").write_bytes(b"not the real weights")
    cfg["weights"] = f"web/weights/{RUN}/best.pt"
    app = web.create_app(cfg, model=model, root=root, dev=True)
    c = TestClient(app)
    ids = [m["run_id"] for m in c.get("/model_info").json()["models"]]
    assert ids == [RUN, other]
    other_info = web.load_model_info({"run_id": other}, root)
    web_models = {m["run_id"]: m for m in c.get("/models").json()}
    assert web_models[other]["threshold"] == other_info["threshold"] and web_models[other]["temporary"] is True
    gray = np.full((200, 200, 3), 120, np.uint8)
    assert c.post("/predict?model=nope", files={"file": ("a.png", png(gray))}).status_code == 404
    r = c.post(f"/predict?model={RUN}", files={"file": ("a.png", png(gray))}).json()
    assert r["run_id"] == RUN


def test_extra_model_with_wrong_sha_not_offered_outside_dev(setup, tmp_path):
    root = tmp_path / "repo"
    (root / "web" / "weights" / "V8_B0_s42").mkdir(parents=True)
    (root / "web" / "weights" / "V8_B0_s42" / "best.pt").write_bytes(b"x")
    (root / "research" / "reports").mkdir(parents=True)
    for f in ("registry.csv", "reports/thresholds_val.json"):
        (root / "research" / f).write_bytes((ROOT / "research" / f).read_bytes())
    assert web.discover_models(root, RUN, dev=False) == {}
    assert list(web.discover_models(root, RUN, dev=True)) == ["V8_B0_s42"]


def test_frozen_runs_reads_manifest_and_missing_file_is_empty(tmp_path):
    assert web.frozen_runs(tmp_path) == {}
    d = tmp_path / "research" / "reports" / "freeze"
    d.mkdir(parents=True)
    (d / "freeze_manifest_draft.json").write_text('{"runs": {"B5_s52": {"best_pt_sha256": "abc"}}}', encoding="utf-8")
    assert web.frozen_runs(tmp_path) == {"B5_s52": "abc"}


# ---------- การแปลงภาพก่อนตรวจ (ภาพสี / ภาพใหญ่) ----------

def test_neu_like_image_is_passed_unchanged():
    img = np.random.default_rng(0).integers(0, 255, (200, 200), dtype=np.uint8)
    img = np.dstack([img, img, img])
    work, plan = web.plan_preprocess(img, "tile")
    assert plan["mode"] == "direct" and plan["gray"] is False and np.array_equal(work, img)


def test_color_image_becomes_gray_three_channel():
    img = np.zeros((200, 200, 3), np.uint8)
    img[..., 2] = 200
    work, plan = web.plan_preprocess(img, "tile")
    assert plan["gray"] is True and plan["mode"] == "direct"
    assert np.array_equal(work[..., 0], work[..., 1]) and np.array_equal(work[..., 1], work[..., 2])


def test_tiles_cover_image_and_boxes_map_back(setup):
    _, model, info = setup
    img = np.full((400, 600, 3), 120, np.uint8)
    dets, _, plan = web.detect_prepared(model, img, info, "cpu", "tile")
    xs, ys = web.tile_origins(600), web.tile_origins(400)
    assert xs[0] == 0 and xs[-1] == 400 and ys[-1] == 200 and plan["tiles"] == len(xs) * len(ys)
    # กรอบตายตัวของโมเดลจำลอง (10,10)-(50,50) ต้องเลื่อนตามจุดเริ่มของแต่ละช่อง และไม่ซ้ำ
    starts = sorted({(d["box"][0] - 10, d["box"][1] - 10) for d in dets if d["class_name"] == "class_B"})
    assert starts == sorted((x, y) for x in xs for y in ys)


def test_resize_maps_boxes_back_to_upload_size(setup):
    _, model, info = setup
    img = np.full((800, 800, 3), 120, np.uint8)
    dets, _, plan = web.detect_prepared(model, img, info, "cpu", "resize")
    assert plan["mode"] == "resize" and plan["scale"] == 0.25
    b = [d for d in dets if d["class_name"] == "class_B"][0]["box"]
    assert b == [40.0, 40.0, 200.0, 200.0]


def test_huge_image_is_downscaled_to_tile_budget():
    img = np.full((3000, 4000, 3), 120, np.uint8)
    work, plan = web.plan_preprocess(img, "tile")
    assert plan["mode"] == "tile" and plan["tiles"] <= web.MAX_TILES and plan["scale"] < 1


def test_compare_runs_every_available_condition(setup, tmp_path, monkeypatch):
    cfg, model, _ = setup
    root = tmp_path / "repo"
    for rid in (RUN, "B1_s42"):
        (root / "web" / "weights" / rid).mkdir(parents=True)
        (root / "web" / "weights" / rid / "best.pt").write_bytes(b"x")
    (root / "research" / "reports").mkdir(parents=True)
    for f in ("registry.csv", "reports/thresholds_val.json"):
        (root / "research" / f).write_bytes((ROOT / "research" / f).read_bytes())
    cfg["weights"] = f"web/weights/{RUN}/best.pt"
    cfg["compare_runs"] = {"YOLO11s": [RUN, "B1_s42", "B2_s62"]}
    app = web.create_app(cfg, model=model, root=root, dev=True)
    app_models = TestClient(app)
    monkeypatch.setattr(web, "load_yolo", lambda *a, **k: model)  # ไม่ต้องมี ultralytics ในการทดสอบ
    gray = np.full((200, 200, 3), 120, np.uint8)
    r = app_models.post("/compare?arch=YOLO11s", files={"file": ("a.png", png(gray))}).json()
    assert [x["run_id"] for x in r["results"]] == [RUN, "B1_s42"] and r["missing"] == ["B2_s62"]
    assert app_models.post("/compare?arch=YOLOv8s", files={"file": ("a.png", png(gray))}).status_code == 404
    assert app_models.post("/predict?prep=zzz", files={"file": ("a.png", png(gray))}).status_code == 400


def test_ground_truth_only_for_validation_images():
    import csv as _csv
    gt = web.load_val_gt()
    rows = list(_csv.DictReader(open(ROOT / "research" / "data" / "dataset_v6" / "manifest.csv", encoding="utf-8")))
    test_shas = {r["sha256"] for r in rows if r["split"] == "test"}
    assert gt and not (test_shas & set(gt))  # ภาพ Test ไม่มีกรอบเฉลยในเว็บเด็ดขาด
    a1 = gt["496a7faddb7fb2c2d2454ba1ac3abcbdb7e0b2d35147015987422027a85d2dda"]  # class_A_0001 (Validation)
    assert a1["image_id"] == "class_A_0001" and len(a1["boxes"]) == 2 and a1["boxes"][0]["class_id"] == 0


class TinyNet:
    """แทน YOLO.model: ชั้น 0–2 + Detect (f = ชั้นที่ส่งเข้า) — พอให้ทดสอบ cam_overlay ได้โดยไม่ต้องมี best.pt"""

    def __init__(self):
        import torch.nn as nn

        class Det(nn.Module):
            f = [1, 2]

        self.mods = nn.ModuleList([nn.Conv2d(3, 4, 3, 2, 1), nn.Conv2d(4, 4, 3, 2, 1), nn.Conv2d(4, 4, 3, 2, 1), Det()])
        self.model = self.mods

    def parameters(self):
        return self.mods.parameters()

    def __call__(self, t):
        x = t
        for m in list(self.mods)[:3]:
            x = m(x)
        return x


class TinyYolo:
    def __init__(self):
        self.model = TinyNet()


def test_cam_overlay_matches_upload_size_and_leaves_no_hooks(setup):
    import base64

    _, _, info = setup
    y = TinyYolo()
    img = np.random.default_rng(0).integers(0, 255, (300, 500, 3), np.uint8)
    url = web.cam_overlay(y, img, info)
    out = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert out.shape == (300, 500, 4)  # สี + ความโปร่งใส ขนาดเท่าภาพที่อัปโหลด
    assert all(not m._forward_hooks for m in y.model.mods)  # ถอด hook ทุกครั้ง ไม่กระทบการตรวจครั้งต่อไป


def test_cam_unknown_model_is_404(setup):
    cfg, model, _ = setup
    r = client(cfg, model, dev=True).post("/cam?model=nope", files={"file": ("a.png", png(np.zeros((200, 200, 3), np.uint8)))})
    assert r.status_code == 404
