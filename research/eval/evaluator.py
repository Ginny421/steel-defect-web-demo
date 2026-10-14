"""Evaluator กลางของโครงงาน (การ์ด B04) ทุก run ต้องให้คะแนนผ่านไฟล์นี้ ไม่ใช้ตัวเลขจาก Ultralytics val โดยตรง

รูปแบบข้อมูล (พิกัดเป็นพิกเซลของภาพขนาดเดิม x1,y1 = มุมซ้ายบน x2,y2 = มุมขวาล่าง):
    GT          image_id,class_id,x1,y1,x2,y2
    prediction  image_id,class_id,x1,y1,x2,y2,confidence
    images      รายชื่อ image_id ทั้งหมดที่ประเมิน รวมภาพที่ไม่มี GT (ภาพปกติ) และภาพที่โมเดลไม่ทำนายอะไรเลย
                ถ้าไม่ระบุ images ภาพปกติที่ไม่มีทั้ง GT และ prediction จะหายไปจากการนับ false alarm

ตัวชี้วัด 2 กลุ่ม แยกกันตาม protocol:
  ไม่ใช้ confidence threshold (ต้องส่ง prediction ที่ confidence ต่ำมาด้วย เช่น conf=0.001 เพื่อให้ได้ PR curve เต็ม)
    - AP50, AP50-95 ต่อคลาส และ mAP = ค่าเฉลี่ยของคลาสที่มี GT
      จับคู่และ interpolate 101 จุดแบบ COCO (pycocotools) ทดสอบแล้วว่าตรงกับ pycocotools
      ต่างจาก COCO เพียงไม่จำกัด 100 กรอบต่อภาพ และไม่มี area range/crowd
  ใช้ confidence threshold (ตรึงจาก Validation เท่านั้น)
    - TP/FP/FN, precision, recall, F1 ต่อคลาสที่ IoU 0.5
    - confusion matrix (รวม background) แบบไม่สนคลาสตอนจับคู่
    - false alarm บนภาพที่ไม่มี GT: FP ต่อภาพ และสัดส่วนภาพที่มีกรอบอย่างน้อย 1 กรอบ

กติกาการจับคู่: เรียง prediction ในภาพตาม confidence มาก->น้อย แต่ละ prediction จับกับ GT คลาสเดียวกัน
ที่ยังไม่ถูกจับและมี IoU สูงสุด (ต้อง >= เกณฑ์) GT หนึ่งกรอบจับได้ครั้งเดียว กรอบทำนายซ้ำจึงเป็น FP
กติกาตัวหารศูนย์: precision = 0 เมื่อไม่มี prediction, recall = 0 เมื่อไม่มี GT, F1 = 0 เมื่อ P+R = 0

Bootstrap: evaluate() รับ image_ids ที่ซ้ำกันได้ (สุ่มแบบใส่คืน) ภาพที่ถูกสุ่มซ้ำจะนับซ้ำตามจำนวนครั้ง

ใช้:
    python research/eval/evaluator.py --gt gt.csv --pred pred.csv --images images.txt --conf 0.25 --out metrics.json
    python research/eval/evaluator.py --dataset research/data/dataset_v6_images --split val --neu-only --write-gt research/eval/gt_val_neu.csv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
REC_THRESHOLDS = np.linspace(0, 1, 101)
GT_COLS = ["image_id", "class_id", "x1", "y1", "x2", "y2"]
PRED_COLS = GT_COLS + ["confidence"]


# ---------------------------------------------------------------- I/O
def read_boxes(path, with_conf):
    cols = PRED_COLS if with_conf else GT_COLS
    out = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        missing = set(cols) - set(rd.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: ขาดคอลัมน์ {sorted(missing)}")
        for r in rd:
            box = [int(r["class_id"])] + [float(r[k]) for k in cols[2:]]
            if box[3] <= box[1] or box[4] <= box[2]:
                raise ValueError(f"{path}: กรอบไม่มีพื้นที่ {r}")
            out[r["image_id"]].append(box)
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}


def write_boxes(path, boxes, with_conf):
    cols = PRED_COLS if with_conf else GT_COLS
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for img in sorted(boxes):
            for b in boxes[img]:
                wr.writerow([img, int(b[0])] + [f"{v:.6g}" for v in b[1:]])


def gt_from_dataset(dataset, split, include_normal=True):
    """อ่าน GT จาก dataset ทางการ (dataset_v6: manifest + label YOLO) คืน (gt, image_ids) พิกัดเป็นพิกเซล"""
    from PIL import Image

    dataset = Path(dataset)
    gt, ids = {}, []
    with open(dataset / "manifest.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] != split or (r["source"] != "NEU-DET" and not include_normal):
                continue
            img = dataset / r["folder"] / split / "images" / r["file"]
            with Image.open(img) as im:
                w, h = im.size
            stem = Path(r["file"]).stem
            ids.append(stem)
            rows = []
            for line in (dataset / r["folder"] / split / "labels" / f"{stem}.txt").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    c, cx, cy, bw, bh = map(float, line.split())
                    rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            if rows:
                gt[stem] = np.array(rows)
    return gt, sorted(ids)


# ---------------------------------------------------------------- core
def iou_matrix(a, b):
    """IoU ระหว่างกรอบ a (N,4) และ b (M,4) รูปแบบ x1,y1,x2,y2"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match(gt, pred, iou_thr, class_aware=True):
    """จับคู่ในภาพเดียว คืน (index ของ GT ที่ prediction แต่ละกรอบจับได้ หรือ -1, ลำดับ prediction มาก->น้อย)
    gt: (N,5) cls,x1..y2  pred: (M,6) cls,x1..y2,conf"""
    order = np.argsort(-pred[:, 5], kind="mergesort") if len(pred) else np.array([], dtype=int)
    matched = np.full(len(pred), -1)
    if len(gt) == 0 or len(pred) == 0:
        return matched, order
    iou = iou_matrix(pred[:, 1:5], gt[:, 1:5])
    taken = np.zeros(len(gt), dtype=bool)
    for p in order:
        cand = iou[p].copy()
        cand[taken] = -1
        if class_aware:
            cand[gt[:, 0] != pred[p, 0]] = -1
        g = int(np.argmax(cand))
        if cand[g] >= iou_thr:
            matched[p] = g
            taken[g] = True
    return matched, order


def average_precision(tp, conf, n_gt):
    """AP แบบ COCO 101 จุด tp/conf คือ prediction ทุกกรอบของคลาสนั้นจากทุกภาพ"""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-conf, kind="mergesort")
    tp = tp[order].astype(np.float64)
    tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
    rec = tpc / n_gt
    prec = tpc / (tpc + fpc)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, REC_THRESHOLDS, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def _safe_div(a, b):
    return a / b if b else 0.0


def evaluate(gt, pred, image_ids=None, num_classes=None, conf=0.25, iou_thr=0.5, cm_iou=0.45):
    """คำนวณตัวชี้วัดทั้งหมด

    gt, pred: dict image_id -> np.array (gt: N,5 / pred: M,6)
    image_ids: รายชื่อภาพที่ประเมิน ซ้ำได้ (bootstrap) ถ้า None ใช้ทุกภาพที่ปรากฏใน gt หรือ pred
    """
    if image_ids is None:
        image_ids = sorted(set(gt) | set(pred))
    empty_gt, empty_pred = np.zeros((0, 5)), np.zeros((0, 6))
    if num_classes is None:
        cls = [int(b[0]) for d in (gt, pred) for v in d.values() for b in v]
        num_classes = max(cls) + 1 if cls else 0
    C = num_classes

    # ---- AP (ไม่ใช้ threshold)
    per_iou_tp = {t: defaultdict(list) for t in IOU_THRESHOLDS}
    confs, n_gt = defaultdict(list), np.zeros(C, dtype=int)
    for img in image_ids:
        g, p = gt.get(img, empty_gt), pred.get(img, empty_pred)
        for c in range(C):
            n_gt[c] += int((g[:, 0] == c).sum())
        for c in np.unique(p[:, 0]).astype(int):
            confs[c].extend(p[p[:, 0] == c, 5])
        for t in IOU_THRESHOLDS:
            m, _ = match(g, p, t)
            for c in np.unique(p[:, 0]).astype(int):
                per_iou_tp[t][c].extend(m[p[:, 0] == c] >= 0)
    ap = np.full((C, len(IOU_THRESHOLDS)), np.nan)
    for c in range(C):
        for k, t in enumerate(IOU_THRESHOLDS):
            ap[c, k] = average_precision(np.array(per_iou_tp[t][c], dtype=bool), np.array(confs[c]), n_gt[c])
    has_gt = n_gt > 0

    # ---- ตัวชี้วัดที่ใช้ threshold
    tp, fp, fn = np.zeros(C, int), np.zeros(C, int), np.zeros(C, int)
    cm = np.zeros((C + 1, C + 1), dtype=int)  # แถว = ทำนาย, คอลัมน์ = จริง, index C = background
    fa_images, fa_boxes, n_bg_images = 0, 0, 0
    for img in image_ids:
        g = gt.get(img, empty_gt)
        p = pred.get(img, empty_pred)
        p = p[p[:, 5] >= conf]
        m, _ = match(g, p, iou_thr)
        for c in range(C):
            is_c = p[:, 0] == c
            tp[c] += int((m[is_c] >= 0).sum())
            fp[c] += int((m[is_c] < 0).sum())
            fn[c] += int((g[:, 0] == c).sum()) - int((m[is_c] >= 0).sum())
        mc, _ = match(g, p, cm_iou, class_aware=False)
        for i, j in enumerate(mc):
            cm[int(p[i, 0]), int(g[j, 0]) if j >= 0 else C] += 1
        for j in set(range(len(g))) - set(mc.tolist()):
            cm[C, int(g[j, 0])] += 1
        if len(g) == 0:
            n_bg_images += 1
            fa_boxes += len(p)
            fa_images += len(p) > 0

    precision = np.array([_safe_div(tp[c], tp[c] + fp[c]) for c in range(C)])
    recall = np.array([_safe_div(tp[c], tp[c] + fn[c]) for c in range(C)])
    f1 = np.array([_safe_div(2 * precision[c] * recall[c], precision[c] + recall[c]) for c in range(C)])
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    P, R = _safe_div(TP, TP + FP), _safe_div(TP, TP + FN)

    return {
        "n_images": len(image_ids),
        "n_gt": n_gt.tolist(),
        "mAP50": float(np.nanmean(ap[has_gt, 0])) if has_gt.any() else float("nan"),
        "mAP50_95": float(np.nanmean(ap[has_gt].mean(1))) if has_gt.any() else float("nan"),
        "AP50": ap[:, 0].tolist(),
        "AP50_95": ap.mean(1).tolist(),
        "conf": conf,
        "iou": iou_thr,
        "TP": tp.tolist(), "FP": fp.tolist(), "FN": fn.tolist(),
        "precision": precision.tolist(), "recall": recall.tolist(), "F1": f1.tolist(),
        "macro_F1": float(f1[has_gt].mean()) if has_gt.any() else float("nan"),
        "micro_precision": P, "micro_recall": R, "micro_F1": _safe_div(2 * P * R, P + R),
        "confusion_matrix": cm.tolist(),
        "background_images": n_bg_images,
        "false_alarm_per_image": _safe_div(fa_boxes, n_bg_images),
        "false_alarm_image_rate": _safe_div(fa_images, n_bg_images),
    }


def bootstrap(gt, pred, image_ids, n=1000, seed=0, **kw):
    """สุ่มภาพแบบใส่คืน n ครั้ง คืน dict ชื่อตัวชี้วัด -> array ค่าจากแต่ละรอบ (เฉพาะตัวชี้วัดที่เป็นตัวเลขเดียว)

    ห้ามใช้รายงานผล: สุ่มรายภาพและไม่จับคู่ policy — ใช้ research/stats/bootstrap.py (สุ่มระดับ sheet_group, D28) แทน"""
    rng = np.random.default_rng(seed)
    ids = np.array(image_ids)
    keys = ["mAP50", "mAP50_95", "macro_F1", "micro_F1", "false_alarm_per_image"]
    out = {k: [] for k in keys}
    for _ in range(n):
        r = evaluate(gt, pred, list(rng.choice(ids, len(ids), replace=True)), **kw)
        for k in keys:
            out[k].append(r[k])
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--images", type=Path, help="ไฟล์รายชื่อ image_id บรรทัดละภาพ")
    ap.add_argument("--num-classes", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dataset", type=Path, help="สร้าง GT จาก dataset (dataset_v6) แทน --gt")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--neu-only", action="store_true", help="ไม่รวมภาพปกติ (Severstal)")
    ap.add_argument("--write-gt", type=Path, help="เขียน GT และรายชื่อภาพ (.images.txt) แล้วจบ")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows: ให้ print ภาษาไทยได้แม้ console เป็น cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    if a.dataset:
        gt, ids = gt_from_dataset(a.dataset, a.split, include_normal=not a.neu_only)
    else:
        gt = read_boxes(a.gt, with_conf=False)
        ids = [l.strip() for l in open(a.images, encoding="utf-8") if l.strip()] if a.images else None
    if a.write_gt:
        write_boxes(a.write_gt, gt, with_conf=False)
        a.write_gt.with_suffix(".images.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        print(f"GT {sum(len(v) for v in gt.values())} กรอบ, {len(ids)} ภาพ -> {a.write_gt}")
        return
    pred = read_boxes(a.pred, with_conf=True)
    unknown = set(pred) - set(ids) if ids else set()
    if unknown:
        raise SystemExit(f"prediction มีภาพที่ไม่อยู่ในรายชื่อ {len(unknown)} ภาพ เช่น {sorted(unknown)[:3]}")
    # กันอัตราแจ้งผิด = 0 เกินจริง: คำทำนายที่ทำเฉพาะภาพ NEU แต่รายชื่อมีภาพปกติ (ภาพไม่มี GT)
    bg = [i for i in (ids or []) if i not in gt]
    meta_p = a.pred.with_suffix(".meta.json")
    if bg and meta_p.exists() and json.loads(meta_p.read_text(encoding="utf-8")).get("neu_only"):
        raise SystemExit(f"prediction ทำนายเฉพาะภาพ NEU (neu_only ใน {meta_p.name}) แต่รายชื่อมีภาพไม่มี GT {len(bg)} ภาพ "
                         "— อัตราแจ้งผิดจะเป็น 0 เกินจริง: ทำนายใหม่โดยไม่ใส่ --neu-only หรือใช้ gt_val_neu")
    if bg and not any(i in pred for i in bg):
        print(f"เตือน: ไม่มีคำทำนายบนภาพไม่มี GT เลย ({len(bg)} ภาพ) — ถ้าไฟล์นี้ทำนายเฉพาะภาพ NEU "
              "อัตราแจ้งผิดจะเป็น 0 เกินจริง", file=sys.stderr)
    res = evaluate(gt, pred, ids, num_classes=a.num_classes, conf=a.conf, iou_thr=a.iou)
    text = json.dumps(res, indent=2, ensure_ascii=False)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(f"mAP50={res['mAP50']:.4f} mAP50-95={res['mAP50_95']:.4f} macro_F1@{a.conf}={res['macro_F1']:.4f} "
          f"FA/img={res['false_alarm_per_image']:.4f} ({res['background_images']} ภาพไม่มี GT)")


if __name__ == "__main__":
    main()
