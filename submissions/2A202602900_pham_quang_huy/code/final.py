"""final.py - Bước 4: chạy TEST đúng MỘT lần cho mỗi seed, ghi predictions/ đúng định dạng eval.py.

Cấu hình (backbone + công thức + phương pháp suy luận) đã chốt hoàn toàn trên VAL trước khi chạy file này.
Với mỗi seed:
  1. dựng model từ checkpoint tốt nhất (chọn theo macro-F1 val lúc train)
  2. VAL: suy luận theo phương pháp đã chọn; khớp nhiệt độ T trên val (nếu --ts)
  3. TEST: suy luận MỘT lần; áp dụng T của val
  4. ghi predictions/<out_id>_seed<k>_test.csv (đã TS nếu --ts), <out_id>_uncal_seed<k>_test.csv (chưa TS),
     <out_id>_seed<k>_val.csv; nhật ký truy cập test vào logs/final/test_access_log.csv
Từ chối ghi đè file test đã có (test chỉ chạy một lần mỗi seed).

    python submissions/<bai>/code/final.py --exp T00 --seeds 0 1 2 --method I00
    python submissions/<bai>/code/final.py --exp F01 --seeds 0 1 2 --method I09 --ts
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
from dataset import load_split  # noqa: E402
from eval import compute_metrics, save_predictions  # noqa: E402
from inference import apply_temperature, ensemble_probs, fit_temperature  # noqa: E402
from inference_exps import model_res  # noqa: E402
from methods import Normalizer, combined_logits, get_method, load_run_model, predict, raw_loader  # noqa: E402

SUB = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True, help="exp_id của (các) model; >1 = ensemble cùng seed")
    ap.add_argument("--out-id", help="tên exp_id trong file dự đoán (mặc định = --exp đầu tiên)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--method", default="I00")
    ap.add_argument("--ts", action="store_true", help="temperature scaling, T khớp trên VAL")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fold", type=int, default=0, help="chỉ cho điểm thưởng nhiều fold (bộ ba file của fold đó)")
    args = ap.parse_args()

    out_id = args.out_id or args.exp[0]
    pred_dir = SUB / "predictions"
    log_dir = SUB / "logs" / "final"
    log_dir.mkdir(parents=True, exist_ok=True)
    _, val_df, test_df = load_split(args.labels, args.fold)
    for seed in args.seeds:
        test_path = pred_dir / f"{out_id}_seed{seed}_test.csv"
        if test_path.exists():
            print(f"[dừng] {test_path} đã tồn tại: test chỉ chạy một lần mỗi seed")
            continue
        val_loader = raw_loader(val_df, args.images, 64, args.workers)
        members = []
        for e in args.exp:
            md, meta = load_run_model(Path(args.runs) / e / f"seed{seed}")
            members.append((md, Normalizer(meta["mean"], meta["std"], "cuda"), meta))
        method = get_method(args.method, *model_res(members[0][2]))

        # VAL
        val_views, val_probs = [], []
        for md, nm, _ in members:
            names_v, y_v, pv, pr = predict(md, nm, val_loader, method)
            val_views.append(pv)
            val_probs.append(pr)
        if len(members) == 1:
            z_val = combined_logits(val_views[0], method.space)
            p_val_uncal = val_probs[0]
        else:
            p_val_uncal = ensemble_probs(val_probs)
            z_val = np.log(np.clip(p_val_uncal, 1e-12, None))
        T = fit_temperature(z_val, y_v) if args.ts else 1.0
        p_val = apply_temperature(z_val, T) if args.ts else p_val_uncal

        # TEST: đúng một lần
        test_loader = raw_loader(test_df, args.images, 64, args.workers)
        t0 = time.strftime("%Y-%m-%d %H:%M:%S")
        test_views, test_probs = [], []
        for md, nm, _ in members:
            names_t, y_t, pv, pr = predict(md, nm, test_loader, method)
            test_views.append(pv)
            test_probs.append(pr)
        if len(members) == 1:
            z_test = combined_logits(test_views[0], method.space)
            p_test_uncal = test_probs[0]
        else:
            p_test_uncal = ensemble_probs(test_probs)
            z_test = np.log(np.clip(p_test_uncal, 1e-12, None))
        p_test = apply_temperature(z_test, T) if args.ts else p_test_uncal

        save_predictions(test_path, names_t, y_t, p_test)
        save_predictions(pred_dir / f"{out_id}_seed{seed}_val.csv", names_v, y_v, p_val)
        if args.ts:
            save_predictions(pred_dir / f"{out_id}_uncal_seed{seed}_test.csv", names_t, y_t, p_test_uncal)
            save_predictions(pred_dir / f"{out_id}_uncal_seed{seed}_val.csv", names_v, y_v, p_val_uncal)
        mv = compute_metrics(y_v, p_val.argmax(1), p_val)
        rec = {"out_id": out_id, "seed": seed, "fold": args.fold, "exps": args.exp, "method": method.code, "K": method.k,
               "ts": args.ts, "T": T, "test_run_at": t0, "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"],
               "val_ece": mv["ece"],
               "checkpoints": [str(Path(args.runs) / e / f"seed{seed}" / "best.pt") for e in args.exp],
               "res": method.res, "crop_pct": method.crop_pct}
        (log_dir / f"{out_id}_seed{seed}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        log = log_dir / "test_access_log.csv"
        pd.DataFrame([{"time": t0, "out_id": out_id, "seed": seed, "fold": args.fold, "method": method.code, "ts": args.ts,
                       "file": test_path.name}]).to_csv(log, mode="a", header=not log.exists(), index=False)
        print(f"[{out_id} seed{seed}] method {method.code} T={T:.3f} | val F1 {mv['macro_f1']:.4f} -> "
              f"đã ghi {test_path.name} (chỉ số test tính bằng eval.py)")
        for md, _, _ in members:
            del md
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
