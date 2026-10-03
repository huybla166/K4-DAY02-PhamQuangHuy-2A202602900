"""dinov2_probe.py - điểm thưởng: linear probe trên DINOv2 ViT-S/14 ĐÓNG BĂNG, so với CNN tinh chỉnh.

- Đặc trưng: token CLS (sau norm) của vit_small_patch14_dinov2.lvd142m ở 224x224 (pos-embed nội suy từ 518),
  ảnh val/test-style: ảnh 256 cắt giữa 224, chuẩn hoá ImageNet. Không augmentation.
- Bộ phân loại: hồi quy logistic đa lớp (sklearn) trên đặc trưng đã chuẩn hoá (StandardScaler).
  C và class_weight chọn bằng 3-fold CV phân tầng TRÊN TRAIN (theo macro-F1); fit lại trên toàn bộ train,
  báo cáo trên VAL. Không dùng test (bonus chỉ so sánh trên val, giống Bước 1).

python submissions/<bai>/code/dinov2_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train  # noqa: E402,F401  (đặt sys.path tới eval.py gốc)
from benchmark import latency_report  # noqa: E402
from dataset import PreloadedEvalLoader, load_split  # noqa: E402
from eval import compute_metrics  # noqa: E402
from model import _local_weights, count_gmacs, count_params  # noqa: E402

SUB = HERE.parent
NAME = "vit_small_patch14_dinov2.lvd142m"


def extract(model, loader):
    feats, ys = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for x, y, _ in loader:
            feats.append(model(x.cuda(non_blocking=True)).float().cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--img", type=int, default=224)
    args = ap.parse_args()

    import timm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    torch.backends.cudnn.benchmark = True
    out = SUB / "logs" / "bonus"
    out.mkdir(parents=True, exist_ok=True)
    kw = {"num_classes": 0, "img_size": args.img}
    local = _local_weights(NAME)
    if local is not None:
        kw["pretrained_cfg_overlay"] = {"file": str(local)}
    model = timm.create_model(NAME, pretrained=True, **kw).cuda().eval()
    cfg = model.pretrained_cfg
    mean, std = tuple(cfg["mean"]), tuple(cfg["std"])
    train_df, val_df, _ = load_split(args.labels, 0)      # CHỈ train + val

    t0 = time.perf_counter()
    tr = PreloadedEvalLoader(train_df, args.images, args.img, mean, std, crop_pct=0.875)
    va = PreloadedEvalLoader(val_df, args.images, args.img, mean, std, crop_pct=0.875)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    Xtr, ytr = extract(model, tr)
    Xva, yva = extract(model, va)
    t_feat = time.perf_counter() - t0
    print(f"đặc trưng: train {Xtr.shape}, val {Xva.shape}; nạp ảnh {t_load:.0f}s, trích đặc trưng {t_feat:.0f}s",
          flush=True)

    # chọn C, class_weight bằng CV trên TRAIN
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    grid = [(c, cw) for c in (0.01, 0.03, 0.1, 0.3, 1.0) for cw in (None, "balanced")]
    cv_rows = []
    for c, cw in grid:
        scores = []
        for a, b in skf.split(Xtr, ytr):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight=cw, max_iter=3000))
            clf.fit(Xtr[a], ytr[a])
            scores.append(f1_score(ytr[b], clf.predict(Xtr[b]), average="macro"))
        cv_rows.append({"C": c, "class_weight": str(cw), "cv_macro_f1_train": float(np.mean(scores)),
                        "cv_std": float(np.std(scores, ddof=1))})
        print(cv_rows[-1], flush=True)
    best = max(cv_rows, key=lambda r: r["cv_macro_f1_train"])
    cw = None if best["class_weight"] == "None" else best["class_weight"]
    t0 = time.perf_counter()
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best["C"], class_weight=cw, max_iter=3000))
    clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    probs = clf.predict_proba(Xva)
    m = compute_metrics(yva, probs.argmax(1), probs)

    lat = latency_report(model, 1, args.img, "amp", iters=200, label=f"{NAME} backbone (đóng băng)")
    lat32 = latency_report(model, 1, args.img, "fp32", iters=200, label=f"{NAME} backbone (đóng băng)")
    res = {"model": NAME, "img_size": args.img, "params_M": count_params(model),
           "gmacs": count_gmacs(model, args.img), "feature_dim": int(Xtr.shape[1]),
           "selection": "3-fold CV phân tầng trên train (macro-F1)", "best": best, "cv": cv_rows,
           "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_bal_acc": m["balanced_acc"],
           "val_ece": m["ece"], "val_f1_per_class": m["f1"].tolist(), "val_recall_per_class": m["recall"].tolist(),
           "time_feature_extract_s": t_feat, "time_fit_s": t_fit,
           "latency_b1_amp_p50_ms": lat["p50_ms"], "latency_b1_amp_p95_ms": lat["p95_ms"],
           "latency_b1_fp32_p50_ms": lat32["p50_ms"], "latency_b1_fp32_p95_ms": lat32["p95_ms"]}
    (out / "dinov2_probe.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.DataFrame(cv_rows).to_csv(out / "dinov2_probe_cv.csv", index=False)
    print(f"DINOv2 linear probe: val macro-F1 {m['macro_f1']:.4f} top1 {m['top1']:.4f} "
          f"(C={best['C']}, class_weight={best['class_weight']})", flush=True)


if __name__ == "__main__":
    main()
