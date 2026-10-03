"""make_results.py - Bước 5: tổng hợp results.xlsx (GUIDE.md mục 6.1) và các biểu đồ tổng hợp.

Mọi số đọc từ log thật: runs/<exp>/seed<k>/summary.json + history.csv, logs/inference/*.csv,
predictions/*.csv (chỉ số test tính lại bằng eval.compute_metrics, cùng định nghĩa với eval.py).

python submissions/<bai>/code/make_results.py --final F01 --final-method I09 --baseline T00
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train  # noqa: E402,F401
import experiments as X  # noqa: E402
from dataset import CLASS_NAMES  # noqa: E402
from eval import compute_metrics, read_pred  # noqa: E402

SUB = HERE.parent


def load_summary(runs: Path, exp: str, seed: int = 0):
    p = runs / exp / f"seed{seed}" / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    cfg = json.loads((runs / exp / f"seed{seed}" / "config.json").read_text(encoding="utf-8"))
    s["_cfg"] = cfg
    return s


def ms(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else float("nan")


def fmt_ms(values, d=4):
    m, s = ms(values)
    return f"{m:.{d}f} ± {s:.{d}f}" if np.isfinite(s) else f"{m:.{d}f}"


def backbones_sheet(runs):
    rows = []
    for exp, (desc, name, _) in X.BACKBONES.items():
        s = load_summary(runs, exp)
        if s is None:
            continue
        c = s["_cfg"]["config"]
        rows.append({"exp_id": exp, "backbone": name, "tag trọng số": s["weight_tag"],
                     "#tham số (M)": s["params_M"], "GMAC": s["gmacs"], "độ phân giải": s["img_size"],
                     "epoch": s["epochs"], "seed": s["seed"], "macro-F1 val": s["val_macro_f1"],
                     "top-1 val": s["val_top1"], "balanced acc val": s["val_bal_acc"],
                     "F1 Chinee val": s["val_f1_per_class"][0], "F1 Snake val": s["val_f1_per_class"][7],
                     "best epoch": s["best_epoch"], "thời gian train/epoch (s)": s["train_time_per_epoch_s"],
                     "thông lượng train (ảnh/s)": s["train_imgs_per_s"],
                     "độ trễ batch-1 p50 (ms)": s.get("latency_b1_fp32_p50_ms"),
                     "độ trễ batch-1 p95 (ms)": s.get("latency_b1_fp32_p95_ms"),
                     "bộ nhớ GPU đỉnh (GiB)": s["peak_mem_GiB"],
                     "micro-batch x tích luỹ": f"{c['batch_size'] // c['accum_steps']} x {c['accum_steps']}",
                     "ảnh curves": Path(s["curve"]).name,
                     "ghi chú": X.BACKBONE_NOTES.get(exp, "công thức nền T00") +
                                ", 1 seed; độ trễ FP32 batch 1, 224, chỉ forward"})
    return pd.DataFrame(rows)


def training_sheet(runs, t00_seeds):
    t00 = [load_summary(runs, "T00", k) for k in t00_seeds]
    t00 = [s for s in t00 if s]
    base_f1 = t00[0]["val_macro_f1"] if t00 else np.nan
    m, sd = ms([s["val_macro_f1"] for s in t00]) if len(t00) > 1 else (base_f1, np.nan)
    base_f1 = m   # so với trung bình T00 qua các seed (chặt hơn so với riêng seed 0)
    rows = []
    tables = [("T", X.TRAINING), ("C", X.COMBOS)]
    for _, table in tables:
        for exp, (desc, axis, diff, _ov) in table.items():
            seeds = t00_seeds if exp == "T00" else [0]
            for k in seeds:
                s = load_summary(runs, exp, k)
                if s is None:
                    continue
                d = s["val_macro_f1"] - base_f1
                verdict = "-" if exp == "T00" else (
                    "tốt hơn (Δ > 2 std)" if np.isfinite(sd) and d > 2 * sd else
                    "kém hơn (Δ < -2 std)" if np.isfinite(sd) and d < -2 * sd else
                    "không phân biệt được (|Δ| ≤ 2 std)" if np.isfinite(sd) else "1 seed")
                rows.append({"exp_id": exp, "backbone": s["backbone"], "trục": axis, "khác T00 ở điểm nào": diff,
                             "seed": k, "macro-F1 val": s["val_macro_f1"], "top-1 val": s["val_top1"],
                             "balanced acc val": s["val_bal_acc"],
                             "Δ macro-F1 so với T00 (TB 3 seed)": d,
                             "std T00 (3 seed)": sd, "kết luận so với nhiễu": verdict,
                             "F1 Chinee val": s["val_f1_per_class"][0], "F1 Snake val": s["val_f1_per_class"][7],
                             "recall Chinee val": s["val_recall_per_class"][0],
                             "recall Snake val": s["val_recall_per_class"][7],
                             "ECE val": s["val_ece"], "best epoch": s["best_epoch"],
                             "thời gian train/epoch (s)": s["train_time_per_epoch_s"],
                             "ảnh curves": Path(s["curve"]).name})
    df = pd.DataFrame(rows)
    if len(t00) > 1:
        agg = {"exp_id": "T00 (mean ± std)", "backbone": t00[0]["backbone"], "trục": "-",
               "khác T00 ở điểm nào": f"{len(t00)} seed: {t00_seeds}",
               "seed": "mean", "macro-F1 val": m, "top-1 val": ms([s["val_top1"] for s in t00])[0],
               "std T00 (3 seed)": sd, "kết luận so với nhiễu": f"macro-F1 {fmt_ms([s['val_macro_f1'] for s in t00])}"}
        df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    return df


def test_metrics(pattern):
    files = sorted(glob.glob(pattern))
    out = []
    for f in files:
        p = read_pred(f)
        m = compute_metrics(p.y_true, p.y_pred, p.probs)
        out.append((p.seed, m))
    return out


def final_sheet(final_id, baseline_id, final_desc, baseline_desc):
    pdir = SUB / "predictions"
    rows, per_class = [], {}
    for exp, desc in ((final_id, final_desc), (baseline_id, baseline_desc)):
        test = dict(test_metrics(str(pdir / f"{exp}_seed*_test.csv")))
        val = dict(test_metrics(str(pdir / f"{exp}_seed*_val.csv")))
        uncal = dict(test_metrics(str(pdir / f"{exp}_uncal_seed*_test.csv")))
        T = {}
        for k in test:
            j = SUB / "logs" / "final" / f"{exp}_seed{k}.json"
            if j.exists():
                T[k] = json.loads(j.read_text(encoding="utf-8"))["T"]
        for k in sorted(test):
            t = test[k]
            rows.append({"exp_id": exp, "cấu hình": desc, "seed": k,
                         "macro-F1 val": val[k]["macro_f1"] if k in val else np.nan,
                         "macro-F1 test": t["macro_f1"], "top-1 test": t["top1"],
                         "balanced acc test": t["balanced_acc"], "ECE test": t["ece"],
                         "ECE test chưa TS": uncal[k]["ece"] if k in uncal else np.nan,
                         "T (khớp trên val)": T.get(k, np.nan),
                         "recall Chinee test": t["recall"][0], "recall Snake test": t["recall"][7],
                         "F1 Chinee test": t["f1"][0], "F1 Snake test": t["f1"][7],
                         "file dự đoán": f"{exp}_seed{k}_test.csv"})
        sub = [r for r in rows if r["exp_id"] == exp]
        agg = {"exp_id": f"{exp} (mean ± std)", "cấu hình": desc, "seed": f"{len(sub)} seed"}
        for col in ("macro-F1 val", "macro-F1 test", "top-1 test", "balanced acc test", "ECE test",
                    "ECE test chưa TS", "recall Chinee test", "recall Snake test", "F1 Chinee test",
                    "F1 Snake test"):
            vals = [r[col] for r in sub if np.isfinite(r[col])]
            agg[col] = fmt_ms(vals) if vals else ""
        rows.append(agg)
        ms_ = [m for _, m in sorted(test.items())]
        per_class[exp] = {key: (np.mean([m[key] for m in ms_], 0), np.std([m[key] for m in ms_], 0, ddof=1))
                          for key in ("precision", "recall", "f1")}
        per_class[exp]["support"] = ms_[0]["support"]
    return pd.DataFrame(rows), per_class


def perclass_sheet(per_class, labels):
    rows = []
    for exp, d in per_class.items():
        for i, c in enumerate(CLASS_NAMES):
            rows.append({"cấu hình": labels.get(exp, exp), "lớp": c, "số ảnh test": int(d["support"][i]),
                         "precision (mean)": d["precision"][0][i], "precision (std)": d["precision"][1][i],
                         "recall (mean)": d["recall"][0][i], "recall (std)": d["recall"][1][i],
                         "F1 (mean)": d["f1"][0][i], "F1 (std)": d["f1"][1][i]})
    return pd.DataFrame(rows)


def style(writer, sheet, df, highlight_col=None, best="max", widths=None):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    ws = writer.sheets[sheet]
    ws.freeze_panes = "B2"
    bold = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="DDEBF7")
    for c in ws[1]:
        c.font = bold
        c.fill = head_fill
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[1].height = 45
    for j, col in enumerate(df.columns, start=1):
        letter = get_column_letter(j)
        width = (widths or {}).get(col, min(max(10, len(str(col)) * 0.9), 28))
        ws.column_dimensions[letter].width = width
        if df[col].dtype.kind == "f":
            for r in range(2, len(df) + 2):
                ws[f"{letter}{r}"].number_format = "0.0000" if df[col].abs().max() < 10 else "0.00"
    if highlight_col and highlight_col in df.columns:
        vals = pd.to_numeric(df[highlight_col], errors="coerce")
        if vals.notna().any():
            idx = int(vals.idxmax() if best == "max" else vals.idxmin())
            fill = PatternFill("solid", fgColor="C6EFCE")
            for c in ws[idx + 2]:
                c.fill = fill
                c.font = bold


def plot_backbones(bb, path: Path):
    """macro-F1 val theo độ trễ batch 1 và theo GMAC (Bước 1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, col, lab in ((axes[0], "độ trễ batch-1 p50 (ms)", "độ trễ p50 batch 1, FP32 (ms)"),
                         (axes[1], "GMAC", "GMAC (224x224)")):
        diag = bb["ghi chú"].str.startswith("chẩn đoán")
        for mask, mk, lab_ in ((~diag, "o", "công thức nền T00"), (diag, "x", "chẩn đoán (B08/B09)")):
            ax.scatter(bb.loc[mask, col], bb.loc[mask, "macro-F1 val"], s=20 + 3 * bb.loc[mask, "#tham số (M)"],
                       marker=mk, alpha=0.75, label=lab_)
        for _, r in bb.iterrows():
            name = r["tag trọng số"].split("_patch")[0] + (" LR×10" if r["exp_id"] == "B09" else "")
            ax.annotate(f"{r['exp_id']} {name}", (r[col], r["macro-F1 val"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.legend(fontsize=7, loc="center right")
        ax.set_xlabel(lab)
        ax.set_ylabel("macro-F1 val")
        ax.grid(alpha=0.3)
    axes[1].set_xscale("log")
    axes[1].set_xticks([0.2, 0.5, 1, 2, 5], ["0,2", "0,5", "1", "2", "5"])
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].set_xlim(0.15, 9)
    fig.suptitle("Bước 1: backbone với công thức nền T00 (1 seed); kích thước điểm ∝ số tham số", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_training(tr, path: Path):
    """Δ macro-F1 val so với T00 seed0 của từng thí nghiệm, kèm dải ±2 std của T00 (3 seed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = tr[(tr["exp_id"].astype(str).str.match(r"^[TC]\d\d$")) & (tr["exp_id"] != "T00") & (tr["seed"] == 0)]
    d = d.sort_values("Δ macro-F1 so với T00 (TB 3 seed)")
    sd = float(pd.to_numeric(tr["std T00 (3 seed)"], errors="coerce").dropna().iloc[0])
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(d) + 1.6))
    colors = ["tab:green" if v > 2 * sd else "tab:red" if v < -2 * sd else "tab:gray"
              for v in d["Δ macro-F1 so với T00 (TB 3 seed)"]]
    ax.barh([f"{r['exp_id']} {r['khác T00 ở điểm nào'][:38]}" for _, r in d.iterrows()],
            d["Δ macro-F1 so với T00 (TB 3 seed)"], color=colors)
    ax.axvspan(-2 * sd, 2 * sd, color="orange", alpha=0.2, label=f"±2 std của T00 (3 seed, std = {sd:.4f})")
    ax.axvline(0, color="k", lw=0.8)
    lo = d["Δ macro-F1 so với T00 (TB 3 seed)"].min()
    if lo < -0.05:   # thí nghiệm kém rất xa (vd. từ đầu): cắt trục, ghi số lên thanh
        ax.set_xlim(-0.05, max(0.02, d["Δ macro-F1 so với T00 (TB 3 seed)"].max() + 0.005))
        for i, v in enumerate(d["Δ macro-F1 so với T00 (TB 3 seed)"]):
            if v < -0.05:
                ax.text(-0.049, i, f"{v:+.3f} (cắt trục)", va="center", fontsize=7, color="white")
    ax.set_xlabel("Δ macro-F1 val so với trung bình T00 (3 seed)")
    ax.set_title("Bước 2: ablation công thức huấn luyện (mỗi thí nghiệm 1 seed)", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_inference(inf, path: Path, sd_seed: float | None = None):
    """Đánh đổi macro-F1 val và độ trễ p50 batch 1 của từng phương pháp suy luận, FP32 và AMP (Bước 3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = inf[inf["exp_id"].str.match(r"^I(0[0-5]|09|10)") & ~inf["exp_id"].str.startswith(("I06", "I07", "I08"))]
    d = d.dropna(subset=["p50_ms", "p50_fp32_ms"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, col, lab in ((axes[0], "p50_fp32_ms", "FP32"), (axes[1], "p50_ms", "AMP (autocast FP16)")):
        sc = ax.scatter(d[col], d["val_macro_f1"], c=np.log2(d["K"].astype(float)), cmap="viridis", s=50,
                        vmin=0, vmax=np.log2(d["K"].max()))
        for _, r in d.iterrows():
            ax.annotate(r["exp_id"], (r[col], r["val_macro_f1"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        i00 = d[d.exp_id == "I00"].iloc[0]
        ax.axhline(i00["val_macro_f1"], color="gray", ls=":", lw=1)
        if sd_seed:
            ax.axhspan(i00["val_macro_f1"] - 2 * sd_seed, i00["val_macro_f1"] + 2 * sd_seed, color="orange", alpha=0.12,
                       label=f"I00 ± 2σ seed ({sd_seed:.4f})")
            ax.legend(fontsize=7, loc="lower right")
        ax.set_xscale("log")
        ticks = [t for t in (5, 7, 10, 15, 20, 30, 50, 70, 100) if d[col].min() * 0.8 <= t <= d[col].max() * 1.25]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(f"độ trễ p50 batch 1, {lab} (ms, thang log)")
        ax.grid(alpha=0.3, which="both")
        ax.set_title(lab, fontsize=9)
    axes[0].set_ylabel("macro-F1 val (F01 seed 0)")
    fig.colorbar(sc, ax=axes, label="log2 K (số view × số model)")
    fig.suptitle("Bước 3: đánh đổi độ chính xác – độ trễ (RTX 3060 Laptop; ảnh 256 đã ở GPU, tính cả sinh view + "
                 "chuẩn hoá + forward + gộp)", fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def bonus_sheet(labels_dir: str):
    """Điểm thưởng: DINOv2 linear probe (val), nhiều fold (test của từng fold), lệch phân phối + thích ứng BN."""
    parts = []
    p = SUB / "logs" / "bonus" / "dinov2_probe.json"
    if p.exists():
        j = json.loads(p.read_text(encoding="utf-8"))
        parts.append(pd.DataFrame([{"mục": "DINOv2 linear probe (đóng băng)", "cấu hình": j["model"],
                                    "tập": "val fold 0", "macro-F1": j["val_macro_f1"], "top-1": j["val_top1"],
                                    "ECE": j["val_ece"], "recall Chinee": j["val_recall_per_class"][0],
                                    "recall Snake": j["val_recall_per_class"][7],
                                    "ghi chú": f"logistic C={j['best']['C']}, class_weight={j['best']['class_weight']} "
                                               f"(chọn bằng CV trên train); p50 b1 AMP "
                                               f"{j['latency_b1_amp_p50_ms']:.2f} ms"}]))
    fold_rows = []
    for f in sorted(glob.glob(str(SUB / "predictions" / "*_f[1-4]_seed*_test.csv"))):
        name = Path(f).name
        fold = int(name.split("_f")[1][0])
        pr = read_pred(f)
        ref = pd.read_csv(Path(labels_dir) / f"test_subset{fold}.csv")
        assert sorted(ref.Filename) == sorted(pr.filenames), f"{name}: tên ảnh khác test_subset{fold}.csv"
        m = compute_metrics(pr.y_true, pr.y_pred, pr.probs)
        fold_rows.append({"mục": "nhiều fold", "cấu hình": name.split("_seed")[0], "tập": f"test fold {fold}",
                          "macro-F1": m["macro_f1"], "top-1": m["top1"], "ECE": m["ece"],
                          "recall Chinee": m["recall"][0], "recall Snake": m["recall"][7], "ghi chú": name})
    if fold_rows:
        parts.append(pd.DataFrame(fold_rows))
    sp = SUB / "logs" / "analysis" / "distribution_shift_val.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        s.insert(0, "mục", np.where(s.get("bn_adapt_macro_f1", pd.Series(np.nan, index=s.index)).notna(),
                                    "thích ứng BN lúc kiểm tra (val nửa B)", "lệch phân phối tự tạo (val)"))
        parts.append(s)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def tidy_latency(lat):
    """Sheet Latency một lược đồ: mỗi dòng = (cấu hình, dtype, batch). Dòng theo phương pháp (đo cả sinh view +
    gộp, có cột amp/fp32 riêng) được tách thành hai dòng AMP và FP32."""
    rows = []
    for _, r in lat.iterrows():
        base = {"cấu hình": r["config"], "GPU": r["gpu"], "độ phân giải": r["img_size"], "K view": r["k_views"],
                "gộp BN": bool(r["fused_bn"]), "torch": r["torch"], "tính tiền xử lý?": r["includes_preprocessing"]}
        if pd.notna(r.get("p50_ms (amp)")):
            scope = "cả phương pháp (sinh view + forward + gộp)"
            for dt in ("amp", "fp32"):
                rows.append({**base, "dtype": dt, "batch": 1, "p50 (ms)": r[f"p50_ms ({dt})"],
                             "p95 (ms)": r[f"p95_ms ({dt})"], "p99 (ms)": r[f"p99_ms ({dt})"],
                             "ảnh/s": 1000.0 / r[f"p50_ms ({dt})"], "phạm vi đo": scope})
            thr = r["images_per_s (batch 32, amp)"]   # đo ở batch 32 (chỉ p50)
            rows.append({**base, "dtype": "amp", "batch": 32, "p50 (ms)": 32000.0 / thr, "p95 (ms)": np.nan,
                         "p99 (ms)": np.nan, "ảnh/s": thr, "phạm vi đo": scope})
        else:
            rows.append({**base, "dtype": r["dtype"], "batch": int(r["batch"]), "p50 (ms)": r["p50_ms"],
                         "p95 (ms)": r["p95_ms"], "p99 (ms)": r["p99_ms"], "ảnh/s": r["images_per_s"],
                         "phạm vi đo": "chỉ forward"})
    cols = ["cấu hình", "GPU", "dtype", "batch", "gộp BN", "p50 (ms)", "p95 (ms)", "p99 (ms)", "ảnh/s",
            "độ phân giải", "K view", "phạm vi đo", "tính tiền xử lý?", "torch"]
    return pd.DataFrame(rows)[cols]


def summary_sheet(bb, tr, inf, fin_df, final_id, baseline_id):
    """Top 10 cấu hình theo macro-F1 val (mọi bước), kèm số seed, độ trễ batch 1 (FP32 và AMP) và chi phí."""
    i00 = inf[inf.exp_id == "I00"].iloc[0] if inf is not None else None
    rows = []
    for _, r in bb.iterrows():
        rows.append({"exp_id": r["exp_id"], "loại": "backbone", "mô tả": f"{r['tag trọng số']} + T00, 1 view",
                     "số seed": 1, "macro-F1 val": r["macro-F1 val"], "top-1 val": r["top-1 val"],
                     "p50 batch 1 FP32 (ms)": r["độ trễ batch-1 p50 (ms)"], "p50 batch 1 AMP (ms)": np.nan,
                     "chi phí": f"{r['GMAC']:.2f} GMAC, {r['#tham số (M)']:.1f}M tham số",
                     "ghi chú": r["ghi chú"].split(",")[0]})
    for _, r in tr.iterrows():
        if "mean" in str(r["exp_id"]) or r["seed"] != 0 or r["exp_id"] == "T00":
            continue
        rows.append({"exp_id": r["exp_id"], "loại": "huấn luyện", "mô tả": f"ConvNeXt-T, {r['khác T00 ở điểm nào']}, 1 view",
                     "số seed": 1, "macro-F1 val": r["macro-F1 val"], "top-1 val": r["top-1 val"],
                     "p50 batch 1 FP32 (ms)": i00["p50_fp32_ms"] if i00 is not None else np.nan,
                     "p50 batch 1 AMP (ms)": i00["p50_ms"] if i00 is not None else np.nan,
                     "chi phí": "như I00 (K = 1)", "ghi chú": r["kết luận so với nhiễu"]})
    if inf is not None:
        for _, r in inf.iterrows():
            if str(r["exp_id"]).startswith(("I06_RAW", "I08_BN")):
                continue
            rows.append({"exp_id": r["exp_id"], "loại": "suy luận", "mô tả": f"{r['method']} [{r['model']}]",
                         "số seed": 1, "macro-F1 val": r["val_macro_f1"], "top-1 val": r["val_top1"],
                         "p50 batch 1 FP32 (ms)": r.get("p50_fp32_ms"), "p50 batch 1 AMP (ms)": r.get("p50_ms"),
                         "chi phí": f"K = {r['K']}", "ghi chú": "trên F01 seed 0"})
    for exp, kind in ((final_id, "chung kết"), (baseline_id, "mốc")):
        sub = fin_df[fin_df["exp_id"] == exp] if len(fin_df) else pd.DataFrame()
        if len(sub):
            v = pd.to_numeric(sub["macro-F1 val"], errors="coerce").dropna()
            rows.append({"exp_id": f"{exp} ({kind})", "loại": kind, "mô tả": sub["cấu hình"].iloc[0],
                         "số seed": len(sub), "macro-F1 val": float(v.mean()), "top-1 val": np.nan,
                         "p50 batch 1 FP32 (ms)": np.nan, "p50 batch 1 AMP (ms)": np.nan, "chi phí": "",
                         "ghi chú": f"TB {len(sub)} seed (std {v.std(ddof=1):.4f}); test ở bảng dưới"})
    df = pd.DataFrame(rows)
    pinned = df[df["loại"].isin(["chung kết", "mốc"])]
    df = df[~df["loại"].isin(["chung kết", "mốc"])].sort_values("macro-F1 val", ascending=False).head(10)
    df.insert(0, "hạng", [str(i) for i in range(1, len(df) + 1)])
    pinned = pinned.copy()
    pinned.insert(0, "hạng", "–")
    return pd.concat([df, pinned], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--final", default="F01")
    ap.add_argument("--final-desc", default="ConvNeXt-T in12k + EMA + Mixup + D4 (= C02), suy luận I04_256 + TS")
    ap.add_argument("--baseline", default="T00")
    ap.add_argument("--baseline-desc", default="T00 (công thức nền) + I00 (1 view)")
    ap.add_argument("--t00-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--labels", default="data/labels")
    args = ap.parse_args()
    runs = Path(args.runs)

    bb = backbones_sheet(runs)
    tr = training_sheet(runs, args.t00_seeds)
    inf_p = SUB / "logs" / "inference" / "inference_val.csv"
    inf = pd.read_csv(inf_p) if inf_p.exists() else None
    lat_p = SUB / "logs" / "inference" / "latency.csv"
    lat = tidy_latency(pd.read_csv(lat_p)) if lat_p.exists() else None
    if inf is not None:
        i00 = inf.loc[inf.exp_id == "I00"].iloc[0]
        inf["relative_cost_fp32_vs_I00"] = inf["p50_fp32_ms"] / i00["p50_fp32_ms"]
    have_final = bool(glob.glob(str(SUB / "predictions" / f"{args.final}_seed*_test.csv")))
    fin_df, per_class = (final_sheet(args.final, args.baseline, args.final_desc, args.baseline_desc)
                         if have_final else (pd.DataFrame(), {}))
    pcs = perclass_sheet(per_class, {args.final: f"{args.final} (chung kết)", args.baseline: f"{args.baseline} (mốc)"}) \
        if per_class else pd.DataFrame()
    summ = summary_sheet(bb, tr, inf, fin_df, args.final, args.baseline)
    bonus = bonus_sheet(args.labels)
    if len(bb):
        plot_backbones(bb, SUB / "figures" / "backbones_tradeoff.png")
    if len(tr) and tr["std T00 (3 seed)"].notna().any():
        plot_training(tr, SUB / "figures" / "training_ablation.png")
    if inf is not None:
        f01 = [load_summary(runs, "F01", k) for k in (0, 1, 2)]
        f01 = [s["val_macro_f1"] for s in f01 if s]
        plot_inference(inf, SUB / "figures" / "inference_tradeoff.png",
                       float(np.std(f01, ddof=1)) if len(f01) > 1 else None)

    out = SUB / "results.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        # Summary đầu tiên: bảng một trang + so sánh chung kết với mốc
        summ.to_excel(w, sheet_name="Summary", index=False)
        style(w, "Summary", summ, "macro-F1 val", widths={"mô tả": 70, "chi phí": 28, "ghi chú": 40})
        if have_final:
            ws = w.sheets["Summary"]
            r0 = len(summ) + 4
            ws.cell(r0, 1, "Chung kết trên TEST (mean ± std qua seed, tính lại từ predictions/ bằng eval.compute_metrics)")
            agg = fin_df[fin_df["exp_id"].astype(str).str.contains("mean")]
            cols = ["exp_id", "cấu hình", "seed", "macro-F1 test", "top-1 test", "ECE test",
                    "recall Chinee test", "recall Snake test"]
            for j, c in enumerate(cols, start=1):
                ws.cell(r0 + 1, j, c)
            for i, (_, r) in enumerate(agg.iterrows(), start=2):
                for j, c in enumerate(cols, start=1):
                    ws.cell(r0 + i, j, r[c])
            from openpyxl.styles import Font, PatternFill
            for c in ws[r0 + 1]:
                c.font = Font(bold=True)
            for c in ws[r0 + 2]:
                c.fill = PatternFill("solid", fgColor="C6EFCE")
            fin = fin_df[fin_df["exp_id"] == args.final]
            base = fin_df[fin_df["exp_id"] == args.baseline]
            if len(fin) and len(base):
                d = fin["macro-F1 test"].mean() - base["macro-F1 test"].mean()
                sd = max(fin["macro-F1 test"].std(ddof=1), base["macro-F1 test"].std(ddof=1))
                rr = r0 + len(agg) + 2
                ws.cell(rr, 1, f"Δ macro-F1 test ({args.final} − {args.baseline})")
                a, b = fin["macro-F1 test"].to_numpy(float), base["macro-F1 test"].to_numpy(float)
                va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
                t = (a.mean() - b.mean()) / np.sqrt(va + vb)
                dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
                try:
                    from scipy import stats
                    pv = f", p = {2 * stats.t.sf(abs(t), dof):.3f}"
                except ImportError:
                    pv = ""
                ws.cell(rr, 4, f"{d:+.4f} (s = {sd:.4f}, Δ/s = {d / sd:.1f}; Welch t = {t:.2f}, df = {dof:.1f}{pv})")
                ws.cell(rr, 5, f"{(fin['top-1 test'].mean() - base['top-1 test'].mean()):+.4f}")
                ws.cell(rr, 6, f"{(fin['ECE test'].mean() - base['ECE test'].mean()):+.4f}")
            fm = HERE / "final_method.json"
            if fm.exists():
                j = json.loads(fm.read_text(encoding="utf-8"))
                ws.cell(r0 + len(agg) + 3, 1, f"Độ trễ cấu hình chung kết ({j['method']}, batch 1, RTX 3060 Laptop): "
                        f"p50 / p95 FP32 = {j['p50_ms_fp32']:.2f} / {j['p95_ms_fp32']:.2f} ms; "
                        f"AMP = {j['p50_ms_amp']:.2f} / {j['p95_ms_amp']:.2f} ms (ngân sách thời gian thực 100 ms)")
            g = SUB / "logs" / "eval" / "grade_I.json"
            if g.exists():
                gj = json.loads(g.read_text(encoding="utf-8"))
                ws.cell(r0 + len(agg) + 4, 1, "eval.py grade (phần I, đề xuất): " + "; ".join(
                    f"{it['code']} {it['points']}/{it['max']}" for it in gj["items"]) + f"; tổng {gj['total']}/20")
        bb.to_excel(w, sheet_name="Backbones", index=False)
        style(w, "Backbones", bb, "macro-F1 val")
        tr.to_excel(w, sheet_name="Training", index=False)
        style(w, "Training", tr, "macro-F1 val", widths={"khác T00 ở điểm nào": 40, "kết luận so với nhiễu": 30})
        if inf is not None:
            inf.to_excel(w, sheet_name="Inference", index=False)
            style(w, "Inference", inf, "val_macro_f1", widths={"method": 45, "model": 40})
        if have_final:
            fin_df.to_excel(w, sheet_name="Final", index=False)
            style(w, "Final", fin_df, None, widths={"cấu hình": 60})
            pcs.to_excel(w, sheet_name="PerClass", index=False)
            style(w, "PerClass", pcs)
        if lat is not None:
            lat.to_excel(w, sheet_name="Latency", index=False)
            style(w, "Latency", lat, None, widths={"config": 50})
        if len(bonus):
            bonus.to_excel(w, sheet_name="Bonus", index=False)
            style(w, "Bonus", bonus, None, widths={"mục": 34, "cấu hình": 30, "ghi chú": 50})
    print(f"đã ghi {out}")
    for name, df in (("Summary", summ), ("Backbones", bb), ("Training", tr), ("Final", fin_df)):
        print(f"\n== {name} ==")
        print(df.to_string(index=False, max_colwidth=40))


if __name__ == "__main__":
    main()
