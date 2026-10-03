"""choose_method.py - chốt phương pháp suy luận cho chung kết, CHỈ dựa trên VAL (logs/inference/inference_val.csv).

Quy tắc (viết ra trước khi chạy Bước 3 đầy đủ):
  - ứng viên: phương pháp dùng MỘT model (1 view, TTA, độ phân giải) — không tính ensemble (I05, cần nhiều model),
    EMA/TS/FP16/BN (I06-I08: không đổi dự đoán hoặc là biến thể triển khai);
  - ràng buộc thời gian thực: p95 batch 1 (AMP hoặc FP32, lấy giá trị nhỏ hơn) <= 50 ms trên GPU đã khai báo;
  - chọn macro-F1 val cao nhất; hoà (chênh < 1e-4) thì chọn p50 nhỏ hơn;
  - luôn áp dụng temperature scaling, T khớp trên val của từng seed (final.py --ts).
Ghi code/final_method.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BUDGET_MS = 50.0


def main():
    df = pd.read_csv(HERE.parent / "logs" / "inference" / "inference_val.csv")
    cand = df[df.exp_id.str.match(r"^I(0[0-4]|09|10)") & ~df.exp_id.str.startswith(("I05", "I06", "I07", "I08"))].copy()
    cand["p95_best_ms"] = cand[["p95_ms", "p95_fp32_ms"]].min(axis=1)
    cand = cand[cand.p95_best_ms <= BUDGET_MS]
    top = cand.val_macro_f1.max()
    pick = cand[cand.val_macro_f1 >= top - 1e-4].sort_values("p50_ms").iloc[0]
    out = {"method": pick.exp_id, "name": pick.method, "val_macro_f1_seed0": float(pick.val_macro_f1),
           "val_macro_f1_I00_seed0": float(df.loc[df.exp_id == "I00", "val_macro_f1"].iloc[0]),
           "p50_ms_amp": float(pick.p50_ms), "p95_ms_amp": float(pick.p95_ms),
           "p50_ms_fp32": float(pick.p50_fp32_ms), "p95_ms_fp32": float(pick.p95_fp32_ms),
           "rule": f"1 model, p95 batch 1 <= {BUDGET_MS} ms, macro-F1 val cao nhất (hoà -> p50 nhỏ hơn), + TS",
           "candidates": cand[["exp_id", "K", "val_macro_f1", "p50_ms", "p95_best_ms"]].to_dict("records")}
    (HERE / "final_method.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(pick.exp_id)


if __name__ == "__main__":
    main()
