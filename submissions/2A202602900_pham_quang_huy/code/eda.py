"""eda.py - Bước 0: kiểm tra chia dữ liệu (README 2.1) và EDA (GUIDE 1.2).

Chạy từ thư mục gốc repo:
    python submissions/<bai>/code/eda.py --images data/images --labels data/labels --out submissions/<bai>
Sinh: figures/eda_class_distribution.png, figures/eda_samples.png, logs/eda.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CLASS_NAMES, NUM_CLASSES, check_split, load_split  # noqa: E402

# Table 1 của bài báo (Olsen et al. 2019)
PAPER_TABLE1 = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-stats", type=int, default=2000, help="số ảnh train lấy mẫu để tính mean/std kênh")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_split(args.labels, 0)
    rep = check_split(train_df, val_df, test_df, args.images)
    all_labels = pd.read_csv(Path(args.labels) / "labels.csv")

    # số ảnh theo lớp: toàn bộ (labels.csv) so với Table 1, và theo từng tập
    full_counts = all_labels["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).tolist()
    table = pd.DataFrame({"class": CLASS_NAMES, "paper_table1": PAPER_TABLE1, "labels_csv": full_counts,
                          "train": rep["per_class"]["train"], "val": rep["per_class"]["val"],
                          "test": rep["per_class"]["test"]})
    table["train+val+test"] = table[["train", "val", "test"]].sum(1)
    table["match_paper"] = table["train+val+test"] == table["paper_table1"]
    print(table.to_string(index=False))
    ratio = max(full_counts) / min(full_counts)
    species = full_counts[:8]

    # labels.csv có đúng tập Filename của hợp ba tập?
    union = set(train_df.Filename) | set(val_df.Filename) | set(test_df.Filename)
    lab_map = dict(zip(all_labels.Filename, all_labels.Label))
    label_consistent = all(lab_map.get(f) == l for df in (train_df, val_df, test_df)
                           for f, l in zip(df.Filename, df.Label))
    mismatches = [{"split": s, "Filename": f, "subset_label": int(l), "subset_class": CLASS_NAMES[int(l)],
                   "labels_csv_label": int(lab_map[f]), "labels_csv_class": CLASS_NAMES[int(lab_map[f])]}
                  for s, df in (("train", train_df), ("val", val_df), ("test", test_df))
                  for f, l in zip(df.Filename, df.Label) if lab_map.get(f) != l]
    print("Nhãn lệch giữa *_subset0.csv và labels.csv:", mismatches)

    # thống kê ảnh: kích thước / mode trên TOÀN BỘ ảnh (đọc header), mean/std kênh trên mẫu train
    sizes, modes = {}, {}
    for f in sorted(union):
        with Image.open(Path(args.images) / f) as im:
            sizes[im.size] = sizes.get(im.size, 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1
    rng = np.random.default_rng(0)
    sample = rng.choice(train_df.Filename.to_numpy(), size=min(args.n_stats, len(train_df)), replace=False)
    acc = np.zeros(3)
    acc2 = np.zeros(3)
    npx = 0
    for f in sample:
        with Image.open(Path(args.images) / f) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        acc += a.reshape(-1, 3).sum(0)
        acc2 += (a.reshape(-1, 3) ** 2).sum(0)
        npx += a.shape[0] * a.shape[1]
    mean = acc / npx
    std = np.sqrt(acc2 / npx - mean ** 2)

    # biểu đồ phân bố lớp
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [2, 1]})
    xs = np.arange(NUM_CLASSES)
    w = 0.27
    for k, (split, color) in enumerate([("train", "#4C72B0"), ("val", "#DD8452"), ("test", "#55A868")]):
        bars = axes[0].bar(xs + (k - 1) * w, table[split], w, label=f"{split} (n={rep['n'][split]})", color=color)
        if split == "train":
            axes[0].bar_label(bars, fontsize=7)
    axes[0].set_xticks(xs, CLASS_NAMES, rotation=25, ha="right")
    axes[0].set_ylabel("số ảnh")
    axes[0].set_yscale("log")
    axes[0].set_title("Fold 0: số ảnh theo lớp và tập (trục log)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].barh(CLASS_NAMES[::-1], full_counts[::-1], color="#8172B2")
    for i, v in enumerate(full_counts[::-1]):
        axes[1].text(v, i, f" {v}", va="center", fontsize=8)
    axes[1].set_title(f"Toàn bộ 17.509 ảnh (lớn/nhỏ = {ratio:.2f}x)")
    axes[1].set_xlabel("số ảnh")
    axes[1].set_xlim(0, max(full_counts) * 1.18)
    fig.tight_layout()
    fig.savefig(out / "figures" / "eda_class_distribution.png", dpi=130)
    plt.close(fig)

    # ảnh mẫu: 5 ảnh mỗi lớp từ train
    n_per = 5
    fig, axes = plt.subplots(NUM_CLASSES, n_per, figsize=(n_per * 1.9, NUM_CLASSES * 1.95))
    for c in range(NUM_CLASSES):
        files = train_df[train_df.Label == c].Filename.sample(n_per, random_state=c).tolist()
        for j, f in enumerate(files):
            with Image.open(Path(args.images) / f) as im:
                axes[c, j].imshow(im.convert("RGB"))
            axes[c, j].set_xticks([])
            axes[c, j].set_yticks([])
            if j == 0:
                axes[c, j].set_ylabel(CLASS_NAMES[c], fontsize=8)
    fig.suptitle("Ảnh mẫu DeepWeeds (train, fold 0), 5 ảnh mỗi lớp", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "eda_samples.png", dpi=110)
    plt.close(fig)

    result = {"split_check": {k: rep[k] for k in ("n", "fraction", "total", "union", "overlap",
                                                     "missing_files", "images_on_disk")},
              "per_class_table": table.to_dict(orient="records"),
              "imbalance_ratio_max_over_min": ratio,
              "species_range": [min(species), max(species)],
              "negatives_fraction": full_counts[8] / sum(full_counts),
              "labels_csv_consistent_with_subsets": label_consistent,
              "label_mismatches_subset_vs_labels_csv": mismatches,
              "all_match_table1": bool(table["match_paper"].all()),
              "image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
              "image_modes": modes,
              "train_sample_channel_mean": mean.tolist(), "train_sample_channel_std": std.tolist(),
              "n_stats_images": int(len(sample))}
    (out / "logs" / "eda.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_class_table"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
