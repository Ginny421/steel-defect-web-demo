"""เว็บสาธิตการตรวจจับตำหนิผิวเหล็ก (การ์ด B11 → A18 · แผน research/docs/web_demo_plan.md)

หลักการ: ผลบนเว็บต้องเป็นจุดใช้งานเดียวกับที่รายงาน
  - โมเดล: run_id ใน config.yaml · sha256 ของ weights ต้องตรงกับ research/registry.csv ไม่งั้นไม่ start
  - threshold: ค่าที่เลือกจาก Validation ใน research/reports/thresholds_val.json (ตายตัว ไม่มี slider)
  - การทำนาย: ค่าเดียวกับ research/eval/predict_to_csv.py (conf 0.001, NMS IoU 0.7, max_det 300, imgsz ของ run)
    แล้วกรองกรอบที่ confidence >= threshold แบบเดียวกับ evaluator.py · แปลงกรอบด้วย rows_from_result ตัวเดียวกัน
  - อ่านภาพด้วย cv2 (BGR) เหมือนที่ Ultralytics อ่านจาก path ตอนทำนาย Val
  - ไม่เก็บไฟล์ภาพที่อัปโหลด (log แค่ metadata)
  - ภาพที่ไม่เหมือน NEU-DET (ภาพเทา 200×200): แปลงก่อนตรวจ ไม่ใช่แค่เตือน (แผน research/docs/web_plan_preprocess_compare_unseen_2026-10-12.md)
      ภาพสี → ระดับเทา 3 ช่อง · ภาพใหญ่/สัดส่วนต่าง → ย่อให้ด้านสั้น = 200 (ค่าเริ่มต้น) หรือตัดเป็นช่อง 200×200 ซ้อนกัน
      (ตัดช่องเหมาะกับภาพแผ่นกว้างที่ความละเอียดเท่าภาพฝึก · ภาพความละเอียดสูงของพื้นที่เดิมตำหนิจะใหญ่เกินช่อง)
      ภาพแบบ NEU-DET ไม่ถูกแปลงเลย ผลจึงเท่าเดิมทุกค่า · ผลบนภาพแหล่งอื่นไม่ได้ประเมินในงานวิจัย (บอกบนหน้าเว็บ)
  - /compare: ภาพเดียวตรวจด้วยทุกเงื่อนไขตาม compare_runs ใน config (seed มัธยฐานของ Val F1 ต่อเงื่อนไข)

รัน (จาก root ของ repo, ultralytics==8.4.170):
    python web/app.py                      # http://127.0.0.1:8000
โหมดพัฒนา (ไม่มี best.pt จริง ใช้ weights อื่นได้ แต่หน้าเว็บขึ้นแถบเตือน): WEB_DEMO_DEV=1
"""

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
sys.path.insert(0, str(ROOT / "research" / "eval"))
sys.path.insert(0, str(ROOT / "research" / "stats"))  # eigen_cam.py (ส่วนเสริม) ใช้ฟังก์ชันเดียวกับงานวิเคราะห์
from predict_to_csv import rows_from_result, sha256  # noqa: E402  ใช้ซ้ำ ไม่เขียนใหม่

PREDICT_KW = {"conf": 0.001, "iou": 0.7, "max_det": 300}  # protocol_v1 ข้อ 6 = predict_to_csv.py
CLASS_NAMES = ["class_A", "class_B", "class_C", "class_D", "class_E", "class_F"]  # NEU-DET-anon (nc 6)
# ชื่อชนิดตำหนิจริง: ภาษาอังกฤษจาก research/data/class_mapping.csv (ชื่อเดียวกับตาราง AP รายคลาสในรายงาน)
# ชื่อไทยและลักษณะตามคู่มือผู้ใช้ (ตาราง tab:classes)
CLASS_TH = {
    "pitted_surface": ("ผิวเป็นหลุม", "หลุมเล็ก ๆ กระจายเป็นบริเวณบนผิว"),
    "inclusion": ("สิ่งเจือปน", "สิ่งแปลกปลอมเป็นจุดหรือแถบฝังในเนื้อเหล็ก"),
    "patches": ("รอยด่าง", "รอยด่างเป็นปื้นบนผิว"),
    "rolled-in_scale": ("สะเก็ดรีดฝัง", "สะเก็ดออกไซด์ที่ถูกรีดฝังลงในผิว"),
    "crazing": ("รอยแตกลายงา", "รอยแตกเป็นเส้นเล็กคล้ายลายงาทั่วผิว"),
    "scratches": ("รอยขีดข่วน", "รอยขีดเป็นเส้นยาวตามแนวการเคลื่อนของแผ่น"),
}
NEU_SIZE = 200
TILE_OVERLAP = 0.2     # ช่องซ้อนกัน 20% เพื่อไม่ให้ตำหนิที่คร่อมรอยต่อหายไป
MAX_TILES = 64         # ภาพใหญ่มากจะถูกย่อก่อนตัด ให้ไม่เกินจำนวนนี้ (CPU ~130 ms ต่อช่อง)
MERGE_IOU = 0.5        # รวมกรอบซ้ำจากช่องที่ซ้อนกัน (คลาสเดียวกัน)
PREP_MODES = ("resize", "tile")  # ค่าเริ่มต้น resize: ทดสอบ 13 ต.ค. ภาพ Val ขยาย 3 เท่า ตัดช่องตรงเฉลย 0/2 · ย่อตรง 1/2 เท่าภาพเดิม

log = logging.getLogger("web_demo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def load_class_info(root=ROOT):
    """รหัสคลาส → ชื่ออังกฤษ ชื่อไทย และลักษณะ · อ่านการจับคู่จาก class_mapping.csv"""
    path = root / "research" / "data" / "class_mapping.csv"
    if not path.is_file():  # ไฟล์คงที่ของโปรเจกต์ (เช่น root ชั่วคราวใน test) → ใช้ของ repo
        path = ROOT / "research" / "data" / "class_mapping.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        m = {r["class_name"]: r["display_name"] for r in csv.DictReader(fh)}
    out = []
    for code in CLASS_NAMES:
        key = m.get(code, code)
        th, desc = CLASS_TH.get(key, ("", ""))
        out.append({"code": code, "en": key.replace("_", " "), "th": th, "desc": desc})
    return out


def load_model_info(cfg, root=ROOT):
    """รวมข้อมูลโมเดลจาก config + registry + thresholds_val.json · หยุดถ้าหา run ไม่เจอ"""
    run_id = cfg["run_id"]
    with open(root / "research" / "registry.csv", newline="", encoding="utf-8") as fh:
        reg = {r["run_id"]: r for r in csv.DictReader(fh)}
    if run_id not in reg:
        raise SystemExit(f"หยุด: ไม่พบ {run_id} ใน research/registry.csv")
    thr = json.loads((root / "research" / "reports" / "thresholds_val.json").read_text(encoding="utf-8"))["runs"]
    if run_id not in thr:
        raise SystemExit(f"หยุด: ไม่พบ threshold ของ {run_id} ใน thresholds_val.json")
    r = reg[run_id]
    return {
        "run_id": run_id,
        "architecture": "YOLOv8s" if r["model"].startswith("yolov8") else "YOLO11s",
        "policy": r["policy"],
        "train_seed": int(r["train_seed"]),
        "imgsz": int(r["imgsz"]),
        "threshold": float(thr[run_id]["threshold"]),
        "threshold_source": "Validation F1 สูงสุด (protocol 5.2)",
        "expected_sha256": r["best_pt_sha256"],
        "temporary": bool(cfg.get("temporary", True)),
        "class_names": CLASS_NAMES,
        "class_info": load_class_info(root),
    }


def load_val_gt(root=ROOT):
    """กรอบเฉลยของภาพชุด Validation (NEU-DET) ค้นด้วย sha256 ของไฟล์
    ใช้ได้เฉพาะ split=val ตาม dataset_v6/manifest.csv · ภาพ Test ไม่มีทางได้กรอบเฉลย (ไม่โหลด GT ของ Test เลย)"""
    man = root / "research" / "data" / "dataset_v6" / "manifest.csv"
    gtf = root / "research" / "eval" / "gt_val.csv"
    if not man.is_file() or not gtf.is_file():
        return {}
    with open(man, newline="", encoding="utf-8") as fh:
        val = {r["sha256"]: r["file"].rsplit(".", 1)[0] for r in csv.DictReader(fh)
               if r["split"] == "val" and r["source"] == "NEU-DET"}
    boxes = {}
    with open(gtf, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            boxes.setdefault(r["image_id"], []).append(
                {"class_id": int(r["class_id"]), "box": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]})
    return {sha: {"image_id": iid, "boxes": boxes.get(iid, [])} for sha, iid in val.items()}


def verify_weights(path, expected, dev):
    """sha256 ของ weights ต้องตรง registry · โหมดพัฒนาให้ผ่านได้แต่ติดธง verified=False"""
    if not Path(path).is_file():
        raise SystemExit(f"หยุด: ไม่พบ weights ที่ {path} — ดู web/README.md ข้อ 1")
    got = sha256(path)
    if got != expected and not dev:
        raise SystemExit(f"หยุด: sha256 ของ {path} = {got[:12]}… ไม่ตรง registry {expected[:12]}…")
    return got, got == expected


def decode_image(data):
    """bytes → ภาพ BGR (แบบเดียวกับ cv2.imread ที่ Ultralytics ใช้) · คืน None ถ้าไม่ใช่ภาพ"""
    if not data:
        return None
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return arr if arr is not None and arr.size else None


def is_color(img):
    b, g, r = (img[..., k].astype(np.int16) for k in range(3))
    return bool(max(np.abs(b - g).max(), np.abs(g - r).max()) > 8)


def neu_like_size(h, w):
    return 0.5 * NEU_SIZE <= min(h, w) and max(h, w) <= 2 * NEU_SIZE and max(h, w) <= 1.5 * min(h, w)


def to_gray3(img):
    """ภาพสี → ระดับเทาแบบเดียวกับ NEU-DET (3 ช่องค่าเท่ากัน)"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.merge([g, g, g])


def tile_origins(n):
    """จุดเริ่มของช่องขนาด NEU_SIZE ตามแกนยาว n · ช่องสุดท้ายชิดขอบ"""
    if n <= NEU_SIZE:
        return [0]
    step = int(NEU_SIZE * (1 - TILE_OVERLAP))
    xs = list(range(0, n - NEU_SIZE + 1, step))
    if xs[-1] + NEU_SIZE < n:
        xs.append(n - NEU_SIZE)
    return xs


def plan_preprocess(img, mode="resize"):
    """คืน (ภาพที่ใช้ตรวจ, แผน) · ภาพแบบ NEU-DET คืนภาพเดิมทุกพิกเซล"""
    h, w = img.shape[:2]
    gray = is_color(img)
    work = to_gray3(img) if gray else img
    plan = {"gray": gray, "mode": "direct", "scale": 1.0, "tiles": 1, "input_size": [w, h]}
    if neu_like_size(h, w) or min(h, w) < NEU_SIZE // 2:
        return work, plan  # ขนาดใกล้ภาพฝึก หรือเล็กเกินจะตัด → ส่งตรง (Ultralytics ขยายเป็น imgsz เอง)
    if mode == "resize":
        sc = NEU_SIZE / min(h, w)
        work = cv2.resize(work, (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        plan.update(mode="resize", scale=sc)
        return work, plan
    sc = 1.0
    while len(tile_origins(round(w * sc))) * len(tile_origins(round(h * sc))) > MAX_TILES:
        sc *= 0.85
    if sc < 1.0:
        work = cv2.resize(work, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    wh, ww = work.shape[:2]
    if min(wh, ww) < NEU_SIZE:  # หลังย่อแล้วด้านสั้นไม่พอหนึ่งช่อง → ใช้แบบย่อทั้งภาพแทน
        return plan_preprocess(img, "resize")
    plan.update(mode="tile", scale=sc, tiles=len(tile_origins(ww)) * len(tile_origins(wh)))
    return work, plan


def merge_boxes(dets):
    """รวมกรอบซ้ำจากช่องที่ซ้อนกัน: คลาสเดียวกันที่ IoU >= MERGE_IOU เก็บกรอบที่ความมั่นใจสูงกว่า"""
    keep = []
    for d in sorted(dets, key=lambda d: -d["confidence"]):
        ax1, ay1, ax2, ay2 = d["box"]
        dup = False
        for k in keep:
            if k["class_id"] != d["class_id"]:
                continue
            bx1, by1, bx2, by2 = k["box"]
            iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
            inter = iw * ih
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            if union > 0 and inter / union >= MERGE_IOU:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


def prep_messages(plan):
    """ข้อความบอกผู้ใช้ว่าแปลงภาพอย่างไร (แทนคำเตือนเดิม) · ภาพแบบ NEU-DET = ไม่มีข้อความ"""
    w, h = plan["input_size"]
    msgs = []
    if plan["gray"]:
        msgs.append("ภาพนี้เป็นภาพสี จึงแปลงเป็นภาพระดับเทาก่อนตรวจ เพราะโมเดลฝึกด้วยภาพระดับเทาของ NEU-DET "
                    "ผลบนภาพจากแหล่งอื่นไม่ได้ประเมินในงานวิจัย")
    if plan["mode"] == "tile":
        extra = f" (ย่อภาพเหลือ {plan['scale'] * 100:.0f}% ก่อนตัด)" if plan["scale"] < 1 else ""
        msgs.append(f"ภาพขนาด {w}×{h} ต่างจากภาพฝึก ({NEU_SIZE}×{NEU_SIZE}) จึงตัดเป็น {plan['tiles']} ช่อง "
                    f"ช่องละ {NEU_SIZE}×{NEU_SIZE} พิกเซลแล้วรวมผล{extra} — โหมดทดลอง ในการทดสอบได้กรอบผิดราว 14 กรอบต่อภาพ "
                    "ผลบนภาพแบบนี้ไม่ได้ประเมินในงานวิจัย")
    elif plan["mode"] == "resize":
        msgs.append(f"ภาพขนาด {w}×{h} ต่างจากภาพฝึก ({NEU_SIZE}×{NEU_SIZE}) จึงย่อให้ด้านสั้นเป็น {NEU_SIZE} พิกเซลก่อนตรวจ "
                    "ตำหนิในภาพจึงเล็กลงตามสัดส่วนเดียวกัน ผลบนภาพแบบนี้ไม่ได้ประเมินในงานวิจัย")
    elif not neu_like_size(h, w):
        msgs.append(f"ภาพขนาด {w}×{h} เล็กกว่าภาพฝึก ({NEU_SIZE}×{NEU_SIZE}) มาก ผลบนภาพแบบนี้ไม่ได้ประเมินในงานวิจัย")
    return msgs


def domain_warnings(img):
    """คงไว้ให้โค้ดเดิมเรียกได้: ข้อความของการแปลงแบบค่าเริ่มต้น"""
    return prep_messages(plan_preprocess(img, "resize")[1])


def detect(model, img, info, device):
    """ทำนายด้วยค่าเดียวกับ predict_to_csv.py แล้วกรองที่ threshold ของ Val (>= แบบ evaluator.py)"""
    res = model.predict(img, imgsz=info["imgsz"], device=device, verbose=False, **PREDICT_KW)[0]
    rows, _ = rows_from_result(res)
    dets = []
    for _, c, x1, y1, x2, y2, s in rows:
        if float(s) >= info["threshold"]:
            dets.append({"class_id": c, "class_name": CLASS_NAMES[c], "confidence": float(s),
                         "box": [float(x1), float(y1), float(x2), float(y2)]})
    dets.sort(key=lambda d: -d["confidence"])
    return dets, round(sum(res.speed.values()), 1)


def detect_prepared(model, img, info, device, mode="resize"):
    """แปลงภาพตามแผนแล้วตรวจ · กรอบคืนเป็นพิกัดของภาพที่อัปโหลด"""
    work, plan = plan_preprocess(img, mode)
    h, w = img.shape[:2]
    if plan["mode"] != "tile":
        dets, ms = detect(model, work, info, device)
    else:
        dets, ms = [], 0.0
        wh, ww = work.shape[:2]
        for y0 in tile_origins(wh):
            for x0 in tile_origins(ww):
                part, t = detect(model, work[y0:y0 + NEU_SIZE, x0:x0 + NEU_SIZE], info, device)
                ms += t
                for d in part:
                    x1, y1, x2, y2 = d["box"]
                    d["box"] = [x1 + x0, y1 + y0, x2 + x0, y2 + y0]
                dets.extend(part)
        dets = merge_boxes(dets)
    if plan["scale"] != 1.0:
        sc = plan["scale"]
        for d in dets:
            x1, y1, x2, y2 = (v / sc for v in d["box"])
            d["box"] = [max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)]
    dets.sort(key=lambda d: -d["confidence"])
    return dets, round(ms, 1), plan


def cam_overlay(model, img, info):
    """ส่วนเสริม: แผนที่ Eigen-CAM (ฟังก์ชันเดียวกับ research/stats/eigen_cam.py) ของ feature 3 ระดับที่เข้า Detect head
    คำนวณบนภาพหลังแปลงแบบย่อทั้งภาพเสมอ (ไม่ทำแบบตัดเป็นช่อง) · ไม่ขึ้นกับคลาส ใช้บรรยายเท่านั้น ไม่ใช่เหตุผลของการตัดสินใจ
    คืน PNG สี+ความโปร่งใส ขนาดเท่าภาพที่อัปโหลด (ย่อถ้าเกิน 1600 px) เป็น data URL"""
    import base64

    import torch
    from eigen_cam import eigen_cam

    work, _ = plan_preprocess(img, "resize")
    h, w = work.shape[:2]
    s = info["imgsz"] / max(h, w)  # 200×200 → 640×640 เหมือนตอนทำนาย · ภาพไม่จัตุรัสปัดเป็นพหุคูณของ 32
    nh, nw = max(32, round(h * s / 32) * 32), max(32, round(w * s / 32) * 32)
    x = cv2.resize(work, (nw, nh), interpolation=cv2.INTER_LINEAR)[..., ::-1].copy()
    net = model.model
    p = next(net.parameters())
    t = (torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0).to(p.device, p.dtype)
    levels = net.model[-1].f  # ชั้นที่ส่งเข้า Detect
    feats = {}
    hooks = [net.model[i].register_forward_hook(lambda m, i_, o, k=i: feats.__setitem__(k, o.detach())) for i in levels]
    try:
        with torch.no_grad():
            net(t)
    finally:
        for hk in hooks:
            hk.remove()
    oh, ow = img.shape[:2]
    sc = min(1.0, 1600 / max(oh, ow))
    oh, ow = max(1, round(oh * sc)), max(1, round(ow * sc))
    per = [cv2.resize(eigen_cam(feats[i][0].float().cpu().numpy()), (ow, oh), interpolation=cv2.INTER_LINEAR) for i in levels]
    cam = np.mean(per, 0)
    cam = cam / cam.max() if cam.max() > 0 else cam
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    rgba = np.dstack([heat, (40 + cam * 160).astype(np.uint8)])  # จุดเด่นทึบกว่า พื้นที่ไม่เด่นยังเห็นภาพเดิม
    ok, buf = cv2.imencode(".png", rgba)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def gray_preview(img, plan):
    """ภาพระดับเทาที่โมเดลเห็น (ไว้แสดงบนหน้าเว็บ) · ไม่มีถ้าไม่ได้แปลงสี"""
    if not plan["gray"]:
        return None
    g = to_gray3(img)
    h, w = g.shape[:2]
    if max(h, w) > 1600:
        sc = 1600 / max(h, w)
        g = cv2.resize(g, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    import base64
    ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def load_yolo(weights, info, device, dev):
    """โหลด Ultralytics + warm-up · ultralytics ต้องเป็น 8.4.170 เท่ากับตอนฝึก (โหมดพัฒนาข้าม)"""
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.4.170" and not dev:
        raise SystemExit(f"หยุด: ultralytics {ultralytics.__version__} — ต้องเป็น 8.4.170 เท่ากับตอนฝึก")
    model = YOLO(str(weights))
    detect(model, np.full((NEU_SIZE, NEU_SIZE, 3), 128, np.uint8), info, device)  # warm-up
    return model


def frozen_runs(root):
    """run ที่อยู่ใน freeze manifest (B14) พร้อม sha256 ของ best.pt · ไม่มีไฟล์ = ว่าง (ถือว่ายังไม่ตรึง)"""
    p = root / "research" / "reports" / "freeze" / "freeze_manifest_draft.json"
    if not p.is_file():
        return {}
    return {k: v.get("best_pt_sha256") for k, v in json.loads(p.read_text(encoding="utf-8")).get("runs", {}).items()}


def discover_models(root, default_id, dev):
    """โมเดลเสริมที่เลือกทดสอบได้ = โฟลเดอร์ web/weights/<run_id>/best.pt ที่ run_id อยู่ใน registry + thresholds_val.json
    ไม่ผ่าน sha256 (นอกโหมดพัฒนา) = ไม่เสนอให้เลือก ไม่ใช่ล้มทั้งแอป"""
    found = {}
    frozen = frozen_runs(root)
    folder = root / "web" / "weights"
    if not folder.is_dir():
        return found
    for d in sorted(folder.iterdir()):
        w = d / "best.pt"
        if d.name == default_id or not w.is_file():
            continue
        try:
            info = load_model_info({"run_id": d.name, "temporary": True}, root)
            # run ที่ตรึงแล้วใน freeze manifest และ sha256 ตรงกับทะเบียน = ไม่ใช่โมเดลชั่วคราว
            info["temporary"] = frozen.get(d.name) != info["expected_sha256"]
            info["sha256"], info["verified"] = verify_weights(w, info["expected_sha256"], dev)
        except SystemExit as e:
            log.warning("ข้ามโมเดล %s: %s", d.name, e)
            continue
        found[d.name] = {"info": info, "weights": w, "model": None}
    return found


def public_info(info):
    return {k: v for k, v in info.items() if k != "expected_sha256"}


def create_app(cfg, model=None, root=ROOT, dev=None):
    dev = os.environ.get("WEB_DEMO_DEV") == "1" if dev is None else dev
    info = load_model_info(cfg, root)
    weights = root / cfg["weights"]
    info["sha256"], info["verified"] = verify_weights(weights, info["expected_sha256"], dev)
    device = str(cfg.get("device", "cpu"))
    if model is None:
        model = load_yolo(weights, info, device, dev)
    info["device"] = device
    info["max_upload_mb"] = int(cfg.get("max_upload_mb", 10))
    max_bytes = info["max_upload_mb"] * 1024 * 1024

    default_id = info["run_id"]
    registry = {default_id: {"info": info, "weights": weights, "model": model}}
    registry.update(discover_models(root, default_id, dev))  # โมเดลเสริมโหลดตอนถูกเลือกครั้งแรก
    for k, v in registry.items():
        v["info"]["device"] = device
        v["info"]["max_upload_mb"] = info["max_upload_mb"]

    def get_model(run_id):
        entry = registry.get(run_id)
        if entry is None:
            raise HTTPException(404, f"ไม่พบโมเดล {run_id}")
        if entry["model"] is None:
            entry["model"] = load_yolo(entry["weights"], entry["info"], device, dev)
        return entry["model"], entry["info"]

    app = FastAPI(title="Steel defect demo", docs_url=None, redoc_url=None)

    @app.get("/")
    def index():
        return FileResponse(WEB / "static" / "index.html")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/model_info")
    def model_info():
        return {**public_info(info), "models": [public_info(v["info"]) for v in registry.values()],
                "demo_runs": {a: r for a, r in (cfg.get("demo_runs") or {}).items() if r in registry},
                "compare_runs": {a: [r for r in rs if r in registry] for a, rs in (cfg.get("compare_runs") or {}).items()},
                "compare_missing": {a: [r for r in rs if r not in registry] for a, rs in (cfg.get("compare_runs") or {}).items()},
                "prep_modes": list(PREP_MODES)}

    @app.get("/models")
    def models():
        return [public_info(v["info"]) for v in registry.values()]

    async def read_image(file):
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"ไฟล์ใหญ่เกิน {max_bytes // (1024 * 1024)} MB")
        img = decode_image(data)
        if img is None:
            raise HTTPException(400, "อ่านไฟล์นี้เป็นภาพไม่ได้ — รองรับ JPG, PNG, BMP")
        return img, data

    def check_prep(prep):
        if prep not in PREP_MODES:
            raise HTTPException(400, f"วิธีแปลงภาพต้องเป็น {', '.join(PREP_MODES)}")

    val_gt = load_val_gt(root)

    def gt_for(data):
        """กรอบเฉลยเมื่อไฟล์ที่อัปโหลดคือภาพ Validation ตัวจริง (sha ตรง) เท่านั้น"""
        import hashlib
        return val_gt.get(hashlib.sha256(data).hexdigest())

    def run_one(run_id, img, prep, t0):
        yolo, minfo = get_model(run_id)
        dets, model_ms, plan = detect_prepared(yolo, img, minfo, device, prep)
        h, w = img.shape[:2]
        return {"run_id": minfo["run_id"], "policy": minfo["policy"], "width": w, "height": h, "detections": dets,
                "count": len(dets), "model_ms": model_ms, "server_ms": round((time.perf_counter() - t0) * 1000, 1),
                "warnings": prep_messages(plan), "preprocess": plan}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), model: str = default_id, prep: str = "resize"):
        t0 = time.perf_counter()
        check_prep(prep)
        get_model(model)  # 404 ก่อนอ่านไฟล์
        img, data = await read_image(file)
        out = run_one(model, img, prep, t0)
        out["view"] = gray_preview(img, out["preprocess"])
        out["gt"] = gt_for(data)
        log.info("predict size=%dx%d boxes=%d model_ms=%.1f run=%s prep=%s", out["width"], out["height"], out["count"],
                 out["model_ms"], model, out["preprocess"]["mode"])
        return JSONResponse(out)  # ไม่เก็บไฟล์ภาพ

    @app.post("/cam")
    async def cam(file: UploadFile = File(...), model: str = default_id):
        """ส่วนเสริม: แผนที่ Eigen-CAM ของโมเดลที่เลือก (ไม่เปลี่ยนผลการตรวจ)"""
        t0 = time.perf_counter()
        yolo, minfo = get_model(model)
        img, _ = await read_image(file)
        out = {"run_id": minfo["run_id"], "cam": cam_overlay(yolo, img, minfo),
               "ms": round((time.perf_counter() - t0) * 1000, 1)}
        log.info("cam size=%dx%d run=%s ms=%.1f", img.shape[1], img.shape[0], model, out["ms"])
        return JSONResponse(out)

    @app.post("/compare")
    async def compare(file: UploadFile = File(...), arch: str = "YOLO11s", prep: str = "resize"):
        """ภาพเดียว ตรวจด้วยทุกเงื่อนไขของโมเดลเดียวกัน (รุ่นตาม compare_runs ที่มีไฟล์ในเครื่อง)"""
        check_prep(prep)
        runs = [r for r in (cfg.get("compare_runs") or {}).get(arch, []) if r in registry]
        if not runs:
            raise HTTPException(404, f"ไม่มีไฟล์โมเดลของ {arch} สำหรับเทียบ — ดู web/README.md")
        img, data = await read_image(file)
        gt = gt_for(data)
        results = []
        for r in runs:
            results.append({**run_one(r, img, prep, time.perf_counter()), "gt": gt})
        log.info("compare arch=%s runs=%d", arch, len(results))
        return JSONResponse({"arch": arch, "results": results,
                             "missing": [r for r in (cfg.get("compare_runs") or {}).get(arch, []) if r not in registry],
                             "view": gray_preview(img, results[0]["preprocess"])})

    return app


def main():
    import uvicorn

    cfg = yaml.safe_load((WEB / "config.yaml").read_text(encoding="utf-8"))
    uvicorn.run(create_app(cfg), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
