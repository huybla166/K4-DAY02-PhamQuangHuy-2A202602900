"""temporal_overlap.py - định lượng mức "cùng buổi chụp" giữa train và val/test của fold 0 (cho phần Hạn chế).

Tên file DeepWeeds là thời điểm chụp `YYYYMMDD-HHMMSS-k.jpg`. Với mỗi ảnh val/test, tìm ảnh train chụp gần nhất
theo thời gian; tỉ lệ ảnh có ảnh train trong vòng vài giây cho biết chia ngẫu nhiên đặt các khung hình liên tiếp
của cùng một cảnh vào cả train lẫn test (điểm test lạc quan so với địa điểm/buổi chụp mới).
Chỉ dùng tên file (không dùng nhãn hay dự đoán test).

python submissions/<bai>/code/temporal_overlap.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SUB = Path(__file__).resolve().parent.parent


def seconds(df: pd.DataFrame) -> np.ndarray:
    s = df.Filename.str.extract(r"(\d{8})-(\d{6})-\d+")
    return pd.to_datetime(s[0] + s[1], format="%Y%m%d%H%M%S").values.astype("datetime64[s]").astype(np.int64)


def main(labels: str = "data/labels"):
    tr = np.sort(seconds(pd.read_csv(Path(labels) / "train_subset0.csv")))
    out = {}
    for split in ("val", "test"):
        t = seconds(pd.read_csv(Path(labels) / f"{split}_subset0.csv"))
        i = np.searchsorted(tr, t)
        d = np.minimum(np.abs(t - tr[np.clip(i - 1, 0, len(tr) - 1)]), np.abs(tr[np.clip(i, 0, len(tr) - 1)] - t))
        out[split] = {f"pct_with_train_image_within_{s}s": float((d <= s).mean() * 100) for s in (0, 5, 10, 30, 60)}
        out[split]["median_gap_s"] = float(np.median(d))
    p = SUB / "logs" / "analysis"
    p.mkdir(parents=True, exist_ok=True)
    (p / "temporal_overlap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
