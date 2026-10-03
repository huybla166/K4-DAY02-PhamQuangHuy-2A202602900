"""photometric_errors.py - phân tích lỗi SAU Bước 4: lỗi có tập trung ở ảnh tối / ám màu / tương phản cao không?

Nhìn lưới ảnh đoán sai (figures/misclassified_test.png) gợi ra hai giả thuyết: (1) ảnh ám màu hồng tím (cân bằng
trắng lệch) và (2) ảnh tối, bóng râm. Kiểm tra định lượng trên dự đoán test của chung kết (mọi seed):
tỉ lệ lỗi ở 10% ảnh "cực đoan" nhất theo từng thống kê so với 90% còn lại. Không ảnh hưởng lựa chọn cấu hình.

python submissions/<bai>/code/photometric_errors.py --pred "submissions/<bai>/predictions/F01_seed*_test.csv"
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train  # noqa: E402,F401  (đặt sys.path tới eval.py gốc)
from eval import read_pred  # noqa: E402


def image_stats(path: Path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    r, g, b = a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()
    return {"magenta": (r + b) / 2 - g, "brightness": a.mean(), "contrast": a.std()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--images", default="data/images")
    args = ap.parse_args()
    files = sorted(glob.glob(args.pred))
    preds = [read_pred(f) for f in files]
    names = preds[0].filenames
    with ThreadPoolExecutor(8) as ex:
        st = list(ex.map(lambda f: image_stats(Path(args.images) / f), names))
    out = {"files": [Path(f).name for f in files], "n_images": len(names)}
    for key, extreme in (("magenta", "high"), ("brightness", "low"), ("contrast", "high")):
        v = np.array([s[key] for s in st])
        sel = v >= np.quantile(v, 0.9) if extreme == "high" else v <= np.quantile(v, 0.1)
        rates = []
        for p in preds:
            assert list(p.filenames) == list(names)
            wrong = p.y_true != p.y_pred
            rates.append((wrong[sel].mean() * 100, wrong[~sel].mean() * 100))
        r = np.array(rates)
        out[f"{key}_{extreme}10pct"] = {"n_images": int(sel.sum()), "error_pct_extreme": r[:, 0].tolist(),
                                        "error_pct_rest": r[:, 1].tolist(),
                                        "mean_error_pct_extreme": float(r[:, 0].mean()),
                                        "mean_error_pct_rest": float(r[:, 1].mean())}
    p = HERE.parent / "logs" / "analysis" / "photometric_errors.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
