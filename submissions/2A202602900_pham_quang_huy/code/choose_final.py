"""choose_final.py - chốt công thức huấn luyện cho chung kết F01, CHỈ dựa trên macro-F1 VAL (Bước 4.1).

Quy tắc (viết ra trước khi có kết quả C01/C02): trong các ứng viên = hai yếu tố đơn vượt 2 std (T13 EMA,
T07 Mixup) và hai kết hợp (C01, C02), chọn cấu hình có macro-F1 val (seed 0) cao nhất. Ghi lại
code/final_choice.json; experiments.py đọc file này để dựng F01.

python submissions/<bai>/code/choose_final.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = ["T13", "T07", "C01", "C02"]


def main(runs: str = "runs"):
    scores = {}
    for e in CANDIDATES:
        s = json.loads((Path(runs) / e / "seed0" / "summary.json").read_text(encoding="utf-8"))
        scores[e] = {"val_macro_f1": s["val_macro_f1"], "val_top1": s["val_top1"], "best_epoch": s["best_epoch"]}
    best = max(CANDIDATES, key=lambda e: scores[e]["val_macro_f1"])
    out = {"base": best, "rule": "macro-F1 val cao nhất (seed 0) trong " + ", ".join(CANDIDATES),
           "candidates": scores}
    (HERE / "final_choice.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
