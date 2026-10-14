"""A16 (งานเสริม): Eigen-CAM ของแบบจำลองที่ตรึงแล้ว บนภาพ Test ชุดเดียวกับ A15 — ใช้ประกอบการอธิบายเท่านั้น ไม่ใช่คำอธิบายเชิงสาเหตุ

วิธี: Eigen-CAM (Muhammad & Yeasin, IJCNN 2020) = principal component แรกของ activation ของชั้นที่เลือก ไม่ใช้ gradient
จึงไม่ขึ้นกับคลาสหรือกรอบใดกรอบหนึ่ง (class-agnostic) — บอกได้แค่ว่า "บริเวณไหนมี activation เด่น" ไม่ได้บอกว่าโมเดลตัดสินเพราะบริเวณนั้น
  - ชั้น: feature map 3 ระดับที่ส่งเข้า Detect head (YOLO11s ชั้น 16 / 19 / 22 = stride 8 / 16 / 32) — คำนวณแยกแต่ละระดับ
    ปรับเป็น 0–1 แล้วเฉลี่ย เพื่อให้ตำหนิเล็ก (P3) และใหญ่ (P5) มีส่วนเท่ากัน · แสดงรายระดับแยกได้ด้วย --per-level
  - เครื่องหมายของ principal component กำหนดไม่ได้ (SVD) → เลือกเครื่องหมายที่ทำให้ projection สัมพันธ์บวกกับค่าเฉลี่ย activation ข้ามช่อง
  - ภาพเข้าแบบเดียวกับตอนทำนาย: NEU 200×200 ย่อขยายเป็น 640×640 (สี่เหลี่ยมจัตุรัสจึงไม่มีขอบเติม) ค่า 0–1 · fp32 · eval mode
ภาพ: 18 ภาพจาก A15 (`error_analysis/figures/figure_index.csv`, เลือกด้วยกติกาตายตัว) · NEU เท่านั้น (D51)
กรอบทำนายใช้ threshold_primary_5_2 ของ run จาก freeze manifest · ตรวจ sha ของ weights กับ freeze ก่อนเสมอ

ใช้ (ต้องมี ultralytics 8.4.170):
  .venv/Scripts/python research/stats/eigen_cam.py --weights B5_s52=<best.pt> [B0_s52=<best.pt>] --images <ภาพ Test NEU> --out research/reports/eigen_cam
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import evaluator  # noqa: E402

FREEZE = ROOT / "reports" / "freeze" / "freeze_manifest_draft.json"
INDEX = ROOT / "reports" / "error_analysis" / "figures" / "figure_index.csv"
CLASSES = ["pitted_surface", "inclusion", "patches", "rolled-in_scale", "crazing", "scratches"]


def eigen_cam(act):
    """act: (C,H,W) numpy → (H,W) projection บน principal component แรก ปรับ 0–1"""
    c, h, w = act.shape
    x = act.reshape(c, -1).T  # (HW, C)
    x = x - x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    proj = (x @ vt[0]).reshape(h, w)
    if np.corrcoef(proj.ravel(), act.mean(0).ravel())[0, 1] < 0:  # แก้เครื่องหมายที่ SVD กำหนดไม่ได้
        proj = -proj
    proj = np.maximum(proj, 0)
    return proj / proj.max() if proj.max() > 0 else proj


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True, help="RUN_ID=path (RUN_ID ต้องอยู่ใน freeze manifest)")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-level", action="store_true")
    a = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")
    import cv2
    import torch
    from ultralytics import YOLO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Tahoma", "Leelawadee UI"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break

    runs = json.load(open(FREEZE, encoding="utf-8"))["runs"]
    models = {}
    for spec in a.weights:
        rid, path = spec.split("=", 1)
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if sha != runs[rid]["best_pt_sha256"]:
            sys.exit(f"หยุด: {rid} sha {sha[:12]}… ไม่ตรง freeze")
        y = YOLO(path)
        net = y.model.float().eval()
        levels = net.model[-1].f  # ชั้นที่ส่งเข้า Detect
        models[rid] = (y, net, levels, runs[rid]["threshold_primary_5_2"])
    with open(INDEX, newline="", encoding="utf-8") as fh:
        picks = [r for r in csv.DictReader(fh) if r["image_id"]]
    files = {p.stem: p for p in a.images.rglob("*.jpg")}
    gt = evaluator.read_boxes(ROOT / "env" / "v11s_test" / "gt_test_neu.csv", False)
    a.out.mkdir(parents=True, exist_ok=True)

    cams = {}  # (rid, image) -> (cam 200x200, per-level list, preds)
    for rid, (y, net, levels, thr) in models.items():
        feats = {}
        hooks = [net.model[i].register_forward_hook(lambda m, i_, o, k=i: feats.__setitem__(k, o.detach())) for i in levels]
        for r in picks:
            img = cv2.imread(str(files[r["image_id"]]))
            x = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)[..., ::-1].copy()
            t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                net(t)
            per = [cv2.resize(eigen_cam(feats[i][0].cpu().numpy()), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) for i in levels]
            cam = np.mean(per, 0)
            cam = cam / cam.max() if cam.max() > 0 else cam
            res = y.predict(str(files[r["image_id"]]), imgsz=640, conf=thr, iou=0.7, max_det=300, verbose=False)[0]
            preds = np.c_[res.boxes.cls.cpu().numpy(), res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()]
            cams[rid, r["image_id"]] = (cam, per, preds)
        for h in hooks:
            h.remove()

    def panel(ax, img, boxes_gt, preds=None, cam=None, title=""):
        ax.imshow(img[..., ::-1])
        if cam is not None:
            ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        for b in boxes_gt:
            ax.add_patch(plt.Rectangle((b[1], b[2]), b[3] - b[1], b[4] - b[2], fill=False, ec="lime", lw=1.6))
        if preds is not None:
            for p in preds:
                ax.add_patch(plt.Rectangle((p[1], p[2]), p[3] - p[1], p[4] - p[2], fill=False, ec="white", lw=1.2, ls="--"))
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    rids = list(models)
    for kind in ("correct", "loc", "miss"):
        rows = [r for r in picks if r["kind"] == kind]
        fig, axes = plt.subplots(len(rows), 1 + len(rids), figsize=(2.3 * (1 + len(rids)), 2.35 * len(rows)))
        axes = np.atleast_2d(axes)
        for i, r in enumerate(rows):
            img = cv2.imread(str(files[r["image_id"]]))
            g = gt.get(r["image_id"], np.zeros((0, 5)))
            panel(axes[i, 0], img, g, title=f"{r['image_id']} · {r['class']} (GT)")
            for j, rid in enumerate(rids):
                cam, _, preds = cams[rid, r["image_id"]]
                panel(axes[i, 1 + j], img, g, preds, cam, f"{rid} Eigen-CAM · ทำนาย {len(preds)} กรอบ")
        fig.suptitle(f"Eigen-CAM (ไม่ขึ้นกับคลาส) · กรณี {kind} จาก A15 · เขียว = GT, ขาวประ = ทำนายที่ threshold ตรึง", fontsize=8)
        fig.tight_layout()
        fig.savefig(a.out / f"F14_eigencam_{kind}.png", dpi=170)
        plt.close(fig)
    if a.per_level:
        r = picks[0]
        img = cv2.imread(str(files[r["image_id"]]))
        fig, axes = plt.subplots(len(rids), 4, figsize=(9, 2.4 * len(rids)))
        axes = np.atleast_2d(axes)
        for j, rid in enumerate(rids):
            cam, per, _ = cams[rid, r["image_id"]]
            for k, (m, lab) in enumerate(zip(per + [cam], ["P3 (stride 8)", "P4 (stride 16)", "P5 (stride 32)", "เฉลี่ย"])):
                panel(axes[j, k], img, [], cam=m, title=f"{rid} · {lab}")
        fig.tight_layout()
        fig.savefig(a.out / "F14_eigencam_levels.png", dpi=170)
        plt.close(fig)
    # สรุปเชิงตัวเลขแบบหยาบ: สัดส่วน "พลัง" ของ CAM ที่ตกในกรอบ GT เทียบสัดส่วนพื้นที่กรอบ (ใช้บรรยาย ไม่ใช่ตัวชี้วัด)
    with open(a.out / "eigencam_in_box.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "image_id", "class", "kind", "cam_mass_in_gt", "gt_area_fraction"])
        for (rid, im), (cam, _, _) in cams.items():
            r = next(x for x in picks if x["image_id"] == im)
            mask = np.zeros_like(cam, bool)
            for b in gt.get(im, np.zeros((0, 5))):
                mask[int(b[2]):int(np.ceil(b[4])), int(b[1]):int(np.ceil(b[3]))] = True
            w.writerow([rid, im, r["class"], r["kind"], round(float(cam[mask].sum() / cam.sum()), 3), round(float(mask.mean()), 3)])
    print("EIGEN_CAM_OK", len(cams))


if __name__ == "__main__":
    main()
