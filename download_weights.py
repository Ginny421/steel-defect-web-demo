"""ดาวน์โหลดไฟล์โมเดล best.pt จาก GitHub Release แล้วตรวจ sha256 เทียบกับ research/registry.csv

ใช้:  python download_weights.py            # ทั้ง 2 รุ่นหลัก
      python download_weights.py B5_s52     # เฉพาะรุ่นที่ระบุ
"""
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "https://github.com/Ginny421/steel-defect-web-demo/releases/download/v1.0.0"
RUNS = ["B5_s52", "V8_B5_s62"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(runs):
    with open(ROOT / "research" / "registry.csv", newline="", encoding="utf-8") as fh:
        expected = {r["run_id"]: r["best_pt_sha256"] for r in csv.DictReader(fh)}
    ok = True
    for run in runs:
        if run not in RUNS:
            print(f"{run}: ไม่มีใน release (มีเฉพาะ {', '.join(RUNS)})")
            ok = False
            continue
        dest = ROOT / "web" / "weights" / run / "best.pt"
        if dest.is_file() and sha256(dest) == expected[run]:
            print(f"{run}: มีไฟล์แล้วและ sha256 ตรง")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        print(f"{run}: กำลังดาวน์โหลด…")
        urllib.request.urlretrieve(f"{BASE}/{run}_best.pt", tmp)
        if sha256(tmp) != expected[run]:
            tmp.unlink()
            print(f"{run}: sha256 ไม่ตรงกับ registry — ลบไฟล์แล้ว ลองใหม่อีกครั้ง")
            ok = False
            continue
        tmp.replace(dest)
        print(f"{run}: เรียบร้อย → {dest.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or RUNS))
