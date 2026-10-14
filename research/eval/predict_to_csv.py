"""แปลงผล model.predict() ของ Ultralytics เป็น CSV ที่ evaluator.py อ่านได้ (การ์ด B04)

ผล: <out>.csv คอลัมน์ image_id,class_id,x1,y1,x2,y2,confidence (พิกเซลของภาพขนาดเดิม) และ <out>.meta.json
(hash ของ weights และ manifest, เวอร์ชัน ultralytics, ค่าที่ใช้ทำนาย, จำนวนภาพ/กรอบ, กรอบที่ถูกตัดเพราะไม่มีพื้นที่)

ค่าตั้งต้นตาม protocol_v1 ข้อ 6: conf = 0.001 (ส่งทุกกรอบเพื่อให้ AP ถูก), NMS IoU = 0.7, max_det = 300,
imgsz = ค่าเดียวกับตอนฝึก (อ่านจาก args.yaml ข้าง weights ถ้าไม่ระบุ --imgsz)

**Test ห้ามทำนายก่อน freeze** (protocol ข้อ 5.5): --split test ต้องระบุ --freeze-signoff ที่ชี้ไปยังไฟล์
research/reports/freeze_signoff.md ไม่งั้นหยุดทันทีโดยไม่โหลดโมเดล — สคริปต์ตรวจแค่ว่า **มีไฟล์อยู่** ไม่ได้ตรวจว่าเซ็นแล้ว
การยืนยันว่าเซ็นครบเป็นหน้าที่ของ freeze-gate (B14) ที่คนตรวจ ห้ามสร้างไฟล์ร่างไว้ก่อนเซ็น

คะแนนจาก CSV นี้ผ่าน evaluator อาจต่างจากที่ `yolo val` รายงานเล็กน้อย เพราะ val ของ Ultralytics ใช้ NMS แบบ multi-label
(ต้องยืนยันใน source 8.4.170) และ half precision บน GPU ส่วน predict() ให้กรอบละคลาสเดียวที่ fp32 — ทุก run ใช้ขั้นตอนเดียวกัน
จึงเทียบ policy ได้ยุติธรรม และตัวเลขที่รายงานมาจาก evaluator (protocol ข้อ 6) ไม่ต้องปรับค่าให้ตรงกับ yolo val

ภาพปกติสำหรับอัตราแจ้งผิด (B10, metric_spec.md) ต้องมาจาก **normal_v2** (D48: คัดกรองแล้ว + JPEG แบบ NEU-DET) ไม่ใช่โฟลเดอร์
normal/ ใน dataset_v6 (PNG 300 ภาพก่อนคัด) → ใช้ --normal-manifest research/data/normal_v2/normal_manifest.csv
--normal-root research/data/normal_v2_images (ภาพอยู่ที่ <normal-root>/<split>/images/<file>) แทน --dataset

ใช้ (ต้องติดตั้ง ultralytics==8.4.170 ตาม research/env/environment_lock.txt):
    python research/eval/predict_to_csv.py --weights research/runs/B0_s42/weights/best.pt \\
        --dataset research/data/dataset_v6_images --split val --neu-only --out research/eval/predictions/B0_s42/val.csv
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

COLS = ["image_id", "class_id", "x1", "y1", "x2", "y2", "confidence"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(dataset, split, neu_only):
    """รายชื่อภาพของ split จาก manifest (ไม่อ่าน label) คืน list ของ Path เรียงตามชื่อ"""
    dataset = Path(dataset)
    with open(dataset / "manifest.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == split and (r["source"] == "NEU-DET" or not neu_only)]
    return sorted((dataset / r["folder"] / split / "images" / r["file"] for r in rows), key=lambda p: p.stem)


def list_normal_images(manifest, root, split):
    """ภาพปกติ normal_v2 ของ split จาก normal_manifest.csv คืน list ของ Path เรียงตามชื่อ · หยุดถ้าไฟล์ไม่มีหรือ sha ไม่ตรง"""
    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == split]
    paths, bad = [], []
    for r in rows:
        p = Path(root) / split / "images" / r["file"]
        if not p.is_file() or sha256(p) != r["jpg_sha256"]:
            bad.append(r["file"])
        paths.append(p)
    if bad:
        sys.exit(f"หยุด: ภาพปกติ {len(bad)} ภาพไม่มีหรือ sha256 ไม่ตรง {manifest} (เช่น {bad[:3]})")
    return sorted(paths, key=lambda p: p.stem)


def _np(x):
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x, dtype=np.float64)


def rows_from_result(result):
    """แปลง Results หนึ่งภาพเป็นแถว CSV คืน (rows, จำนวนกรอบที่ตัดเพราะไม่มีพื้นที่)
    ใช้แค่ result.path และ result.boxes.{xyxy, cls, conf} (พิกัดพิกเซลของภาพเดิม) จึงทดสอบด้วยวัตถุจำลองได้"""
    image_id = Path(result.path).stem
    b = result.boxes
    xyxy, cls, conf = _np(b.xyxy).reshape(-1, 4), _np(b.cls).reshape(-1), _np(b.conf).reshape(-1)
    rows, dropped = [], 0
    for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
        if x2 <= x1 or y2 <= y1:
            dropped += 1
            continue
        rows.append([image_id, int(c), f"{x1:.6g}", f"{y1:.6g}", f"{x2:.6g}", f"{y2:.6g}", f"{s:.6g}"])
    return rows, dropped


def imgsz_from_args(weights):
    """อ่าน imgsz จาก args.yaml ของ run (โฟลเดอร์แม่ของ weights/) ถ้าไม่มีคืน None"""
    p = Path(weights).resolve().parent.parent / "args.yaml"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("imgsz:"):
            v = line.split(":", 1)[1].strip().strip("[]").replace(",", " ").split()
            if len(v) == 1 or (len(v) == 2 and v[0] == v[1]):
                return int(v[0])
            sys.exit(f"หยุด: imgsz ใน {p} เป็น {v} (ไม่ใช่สี่เหลี่ยมจัตุรัส) — ระบุ --imgsz เอง")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, help="dataset_v6 (ภาพ NEU) — ไม่ใช้เมื่อระบุ --normal-manifest")
    ap.add_argument("--normal-manifest", type=Path, help="normal_v2/normal_manifest.csv: ทำนายเฉพาะภาพปกติ normal_v2")
    ap.add_argument("--normal-root", type=Path, help="โฟลเดอร์ภาพ normal_v2 (<root>/<split>/images)")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--neu-only", action="store_true", help="เฉพาะภาพ NEU-DET (ใช้คิด mAP ของ B0–B5/B_default)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, help="ค่าเริ่ม = imgsz ใน args.yaml ของ run")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--freeze-signoff", type=Path, help="จำเป็นเมื่อ --split test: path ของ freeze_signoff.md (B14)")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows: ให้ print ภาษาไทยได้แม้ console เป็น cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    if a.split == "test" and not (a.freeze_signoff and a.freeze_signoff.is_file()):
        sys.exit("หยุด: ห้ามทำนายบน Test ก่อน freeze (protocol ข้อ 5.5) — ต้องมี --freeze-signoff ชี้ไปยัง "
                 "research/reports/freeze_signoff.md (สคริปต์ตรวจแค่ว่ามีไฟล์ การเซ็นครบตรวจที่ B14)")
    imgsz = a.imgsz or imgsz_from_args(a.weights)
    if imgsz is None:
        sys.exit("หยุด: ไม่พบ imgsz ใน args.yaml ของ run — ระบุ --imgsz ให้ตรงกับตอนฝึก")
    if a.normal_manifest:
        if not a.normal_root or a.dataset or a.neu_only:
            sys.exit("หยุด: --normal-manifest ต้องใช้คู่กับ --normal-root และไม่ใช้ --dataset/--neu-only")
        images = list_normal_images(a.normal_manifest, a.normal_root, a.split)
    elif a.dataset:
        images = list_images(a.dataset, a.split, a.neu_only)
    else:
        sys.exit("หยุด: ต้องระบุ --dataset หรือ --normal-manifest")
    if not images:
        sys.exit(f"ไม่มีภาพใน split {a.split}")

    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(a.weights))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n_boxes = n_dropped = 0
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(COLS)
        for i in range(0, len(images), a.batch):
            for r in model.predict([str(p) for p in images[i:i + a.batch]], imgsz=imgsz, conf=a.conf, iou=a.iou,
                                   max_det=a.max_det, device=a.device, verbose=False):
                rows, d = rows_from_result(r)
                wr.writerows(rows)
                n_boxes += len(rows)
                n_dropped += d
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": str(a.weights), "weights_sha256": sha256(a.weights),
        "dataset": str(a.dataset or a.normal_root),
        "manifest_sha256": sha256(a.normal_manifest if a.normal_manifest else a.dataset / "manifest.csv"),
        "normal_v2": bool(a.normal_manifest),
        "split": a.split, "neu_only": a.neu_only, "n_images": len(images), "n_boxes": n_boxes,
        "dropped_zero_area_boxes": n_dropped,
        "imgsz": imgsz, "conf": a.conf, "iou": a.iou, "max_det": a.max_det,
        "ultralytics": ultralytics.__version__,
        "freeze_signoff": str(a.freeze_signoff) if a.freeze_signoff else None,
    }
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(images)} ภาพ, {n_boxes} กรอบ -> {a.out}" + (f" (ตัดกรอบไม่มีพื้นที่ {n_dropped})" if n_dropped else ""))
    if n_dropped:
        print("เตือน: มีกรอบไม่มีพื้นที่ถูกตัด — จำนวนบันทึกใน meta.json ต้องรายงานถ้าไม่ใช่ 0")


if __name__ == "__main__":
    main()
