"""inference_exps.py - Bước 3: so sánh phương pháp suy luận trên VAL (không huấn luyện lại).

Chạy (từ thư mục gốc repo):
    python submissions/<bai>/code/inference_exps.py --main runs/C01/seed0 \
        --ensemble runs/B03/seed0 runs/B05/seed0 --ema-run runs/T13/seed0 --bn-run runs/B01/seed0
Sinh: logs/inference/inference_val.csv, latency.csv, ts.json; figures/inference_tradeoff.png,
      figures/reliability_val.png. Chỉ dùng VAL (test không được mở ở bước này).

Quy ước: độ chính xác đo với AMP (autocast FP16). Độ trễ đo cho CẢ phương pháp (sinh view + chuẩn hoá +
K lượt forward xếp chung một batch + gộp), đầu vào ảnh 256 đã nằm trên GPU (không tính đọc/giải mã ảnh),
warmup 10, 200 lần đo, có torch.cuda.synchronize; batch 1 ở AMP và FP32, thông lượng ở batch 32 (AMP).
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
from benchmark import bench, latency_report  # noqa: E402
from dataset import load_split  # noqa: E402
from eval import compute_metrics  # noqa: E402
from inference import apply_temperature, ensemble_probs, fit_temperature, fuse_conv_bn  # noqa: E402
from methods import (Method, Normalizer, build_methods, combined_logits, get_method,  # noqa: E402
                     load_run_model, predict, raw_loader, run_views)

SUB = HERE.parent


def model_res(meta) -> tuple[int, float]:
    c = meta["config"]
    return int(c.get("img_size", 224)), float(c.get("eval_crop_pct", 0.875))


def metrics_row(code, name, probs, y, k, model_desc, extra=None):
    m = compute_metrics(y, probs.argmax(1), probs)
    row = {"exp_id": code, "method": name, "model": model_desc, "K": k, "val_macro_f1": m["macro_f1"],
           "val_top1": m["top1"], "val_bal_acc": m["balanced_acc"], "val_ece": m["ece"], "val_nll": m["nll"],
           "val_f1_chinee": m["f1"][0], "val_f1_snake": m["f1"][7],
           "val_recall_chinee": m["recall"][0], "val_recall_snake": m["recall"][7]}
    row.update(extra or {})
    return row


def members_latency(members, batch: int, iters: int, amp: bool):
    """members: list (model, normalizer, method). Đo cả chuỗi: mọi model, mọi view, gộp xác suất."""
    x = torch.rand(batch, 3, 256, 256, device="cuda")

    def fn():
        ps = []
        for md, nm, m in members:
            outs = run_views(md, nm, x, m, amp=amp)
            if m.space == "prob":
                ps.append(torch.stack([o.softmax(1) for o in outs]).mean(0))
            else:
                ps.append(torch.stack(outs).mean(0).softmax(1))
        torch.stack(ps).mean(0)

    return bench(fn, warmup=10, iters=iters, sync=torch.cuda.synchronize)


def latency_cols(members, iters):
    a1 = members_latency(members, 1, iters, amp=True)
    f1 = members_latency(members, 1, iters, amp=False)
    a32 = members_latency(members, 32, max(50, iters // 4), amp=True)
    return {"p50_ms": a1["p50"], "p95_ms": a1["p95"], "p99_ms": a1["p99"],
            "p50_fp32_ms": f1["p50"], "p95_fp32_ms": f1["p95"], "p99_fp32_ms": f1["p99"],
            "throughput_img_s": 32 / (a32["p50"] / 1000), "latency_dtype": "amp (cột fp32 riêng)"}


def tta_flip_analysis(base_probs, tta_probs, y):
    b, t = base_probs.argmax(1), tta_probs.argmax(1)
    return {"tta_changed": int((b != t).sum()), "tta_wrong_to_right": int(((b != y) & (t == y)).sum()),
            "tta_right_to_wrong": int(((b == y) & (t != y)).sum())}


def cross_fit_ece(logits, y, n_splits=2, seed=0):
    """ECE trung thực hơn: khớp T trên một nửa val, đo ECE trên nửa còn lại (2 chiều)."""
    rng = np.random.default_rng(seed)
    halves = np.array_split(rng.permutation(len(y)), n_splits)
    tot, Ts = 0.0, []
    for i in range(n_splits):
        te = halves[i]
        fit = np.concatenate([halves[j] for j in range(n_splits) if j != i])
        T = fit_temperature(logits[fit], y[fit])
        p = apply_temperature(logits[te], T)
        tot += compute_metrics(y[te], p.argmax(1), p)["ece"] * len(te)
        Ts.append(T)
    return float(tot / len(y)), Ts


def reliability(ax, probs, y, title, bins=15):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    accs = [correct[idx == b].mean() if (idx == b).any() else 0 for b in range(bins)]
    confs = [conf[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)]
    centers = (np.arange(bins) + 0.5) / bins
    ax.bar(centers, accs, width=1 / bins, edgecolor="k", alpha=0.7, label="accuracy theo bin")
    ax.plot(centers, confs, "o", ms=3, color="orange", label="độ tin cậy TB của bin")
    ax.plot([0, 1], [0, 1], "r--", label="hiệu chuẩn hoàn hảo")
    ax.set_xlabel("độ tin cậy (max softmax)")
    ax.set_ylabel("accuracy")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="run_dir của model chính (checkpoint tốt nhất trên val)")
    ap.add_argument("--ensemble", nargs="*", default=[], help="run_dir của các model thêm vào ensemble")
    ap.add_argument("--ema-run", help="run_dir có best.pt (EMA) và best_raw.pt (không EMA)")
    ap.add_argument("--bn-run", help="run_dir của một CNN có BatchNorm để thử gộp BN")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--limit", type=int, help="chỉ N ảnh val đầu tiên (chạy thử)")
    ap.add_argument("--out", help="thư mục kết quả (mặc định logs/inference)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out) if args.out else SUB / "logs" / "inference"
    out.mkdir(parents=True, exist_ok=True)
    figs = out if args.out else SUB / "figures"
    torch.backends.cudnn.benchmark = True
    _, val_df, _ = load_split(args.labels, 0)          # CHỈ val
    if args.limit:
        val_df = val_df.head(args.limit)
    loader = raw_loader(val_df, args.images, 64, args.workers)
    gpu = torch.cuda.get_device_name(0)

    main_dir = Path(args.main)
    model, meta = load_run_model(main_dir)
    res, crop = model_res(meta)
    methods = build_methods(res, crop)
    norm = Normalizer(meta["mean"], meta["std"], "cuda")
    main_desc = f"{meta['config']['exp_id']} seed{meta['config']['seed']} ({meta['weight_tag']}, {res}px)"
    rows, lat_rows, saved = [], [], {}

    def add_latency_row(code, desc, k, r, img, extra=None):
        lat_rows.append({"config": f"{code} | {desc}", "gpu": gpu, "batch": 1, "k_views": k, "img_size": img,
                         "fused_bn": False, "p50_ms (amp)": r["p50_ms"], "p95_ms (amp)": r["p95_ms"],
                         "p99_ms (amp)": r["p99_ms"], "p50_ms (fp32)": r["p50_fp32_ms"],
                         "p95_ms (fp32)": r["p95_fp32_ms"], "p99_ms (fp32)": r["p99_fp32_ms"],
                         "images_per_s (batch 32, amp)": r["throughput_img_s"], "torch": torch.__version__,
                         "includes_preprocessing": "không (ảnh đã ở GPU)", **(extra or {})})

    # 1) các phương pháp trên model chính
    for m in methods:
        t0 = time.perf_counter()
        names, y, per_view, probs = predict(model, norm, loader, m)
        saved[m.code] = (per_view, probs)
        extra = {"views": m.views, "res": m.res, "space": m.space}
        if m.code == "I00":   # I00 phải trùng logit đã lưu lúc train (kiểm tra pipeline suy luận)
            ref = np.load(main_dir / "val_logits.npy")[:len(names)]
            assert pd.read_csv(main_dir / "val_files.csv").Filename.tolist()[:len(names)] == names,                 "thứ tự file khác lúc train"
            extra["max_abs_diff_vs_train_logits"] = float(np.abs(ref - per_view[0]).max())
            extra["argmax_agree_vs_train"] = float((ref.argmax(1) == per_view[0].argmax(1)).mean())
            print(f"I00 so với val_logits lúc train: |Δ| max {extra['max_abs_diff_vs_train_logits']:.4f}, "
                  f"argmax trùng {extra['argmax_agree_vs_train']:.4f}", flush=True)
        if m.k > 1 or m.code.startswith("I04"):
            extra.update(tta_flip_analysis(saved["I00"][1], probs, y))
        lat = latency_cols([(model, norm, m)], args.iters)
        extra.update(lat)
        extra["eval_time_s"] = time.perf_counter() - t0
        rows.append(metrics_row(m.code, m.name, probs, y, m.k, main_desc, extra))
        add_latency_row(m.code, main_desc, m.k, lat, m.res)
        print(f"{m.code:8s} K={m.k:2d} F1 {rows[-1]['val_macro_f1']:.4f} top1 {rows[-1]['val_top1']:.4f} "
              f"ECE {rows[-1]['val_ece']:.4f} | p50 amp {lat['p50_ms']:.2f} / fp32 {lat['p50_fp32_ms']:.2f} ms",
              flush=True)

    def row_of(code):
        return next(r for r in rows if r["exp_id"] == code)

    # 2) I03: gộp xác suất vs gộp logit đã có trong I01/I01L, I09/I09L.
    # 3) I07: temperature scaling trên VAL cho I00 và cho phương pháp TTA tốt nhất
    ts = {}
    best_tta = max((r for r in rows if r["K"] > 1), key=lambda r: r["val_macro_f1"])["exp_id"]
    for code in dict.fromkeys(["I00", best_tta]):
        per_view, probs = saved[code]
        m = get_method(code, res, crop)
        z = combined_logits(per_view, m.space)
        T = fit_temperature(z, y)
        p_cal = apply_temperature(z, T)
        ece_cv, Ts = cross_fit_ece(z, y)
        before, after = compute_metrics(y, probs.argmax(1), probs), compute_metrics(y, p_cal.argmax(1), p_cal)
        ts[code] = {"T": T, "ece_before": before["ece"], "ece_after_same_val": after["ece"],
                    "ece_after_crossfit_2fold": ece_cv, "T_crossfit": Ts, "nll_before": before["nll"],
                    "nll_after": after["nll"],
                    "argmax_unchanged": bool((p_cal.argmax(1) == probs.argmax(1)).all())}
        r0 = row_of(code)
        rows.append(metrics_row(f"I07_{code}", f"{code} + temperature scaling (T={T:.3f}, khớp trên val)",
                                p_cal, y, m.k, main_desc,
                                {"T": T, "val_ece_crossfit": ece_cv,
                                 **{k: r0[k] for k in ("p50_ms", "p95_ms", "p99_ms", "p50_fp32_ms", "p95_fp32_ms",
                                                       "p99_fp32_ms", "throughput_img_s")}}))
        saved[f"I07_{code}"] = (None, p_cal)
        print(f"I07 {code}: T={T:.3f}, ECE {before['ece']:.4f} -> {after['ece']:.4f} "
              f"(cross-fit 2 nửa val: {ece_cv:.4f})", flush=True)
    (out / "ts.json").write_text(json.dumps(ts, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    reliability(axes[0], saved["I00"][1], y, f"I00 trước TS: ECE {ts['I00']['ece_before']:.4f}")
    reliability(axes[1], saved["I07_I00"][1], y,
                f"I00 sau TS (T={ts['I00']['T']:.2f}): ECE {ts['I00']['ece_after_same_val']:.4f}")
    fig.suptitle(f"Reliability diagram trên val, {main_desc}", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "reliability_val.png", dpi=130)
    plt.close(fig)

    # 4) I08: FP32 thuần vs FP16 (model.half()) vs AMP: độ chính xác + độ trễ forward
    m0 = get_method("I00", res, crop)
    half = load_run_model(main_dir)[0].half()
    _, _, pv16, p16 = predict(half, norm, loader, m0, amp=False, dtype=torch.float16)
    _, _, pv32, p32 = predict(model, norm, loader, m0, amp=False)
    ref32 = pv32[0]
    for code, name, probs_, dt, pv in (("I08_FP32", "1 view FP32 thuần", p32, "fp32", pv32[0]),
                                       ("I08_FP16", "1 view FP16 (model.half())", p16, "fp16", pv16[0]),
                                       ("I08_AMP", "1 view AMP autocast FP16", saved["I00"][1], "amp",
                                        saved["I00"][0][0])):
        lb = {b: latency_report(model, b, res, dtype=dt, iters=args.iters if b == 1 else max(50, args.iters // 4),
                                label=f"{code} | {main_desc}") for b in (1, 8, 32)}
        rows.append(metrics_row(code, name, probs_, y, 1, main_desc,
                                {"p50_ms": lb[1]["p50_ms"], "p95_ms": lb[1]["p95_ms"], "p99_ms": lb[1]["p99_ms"],
                                 "throughput_img_s": lb[32]["images_per_s"], "latency_dtype": dt, "res": res,
                                 "max_abs_logit_diff_vs_fp32": float(np.abs(pv - ref32).max()),
                                 "argmax_agree_vs_fp32": float((pv.argmax(1) == ref32.argmax(1)).mean()),
                                 "note": "độ trễ chỉ forward (latency_report)"}))
        for b, r in lb.items():
            lat_rows.append({"config": r["config"], "gpu": gpu, "dtype": dt, "batch": b, "k_views": 1,
                             "img_size": res, "fused_bn": False, "p50_ms": r["p50_ms"], "p95_ms": r["p95_ms"],
                             "p99_ms": r["p99_ms"], "images_per_s": r["images_per_s"], "torch": torch.__version__,
                             "includes_preprocessing": "không (chỉ forward)"})
    del half
    torch.cuda.empty_cache()

    # 5) I05: ensemble (trung bình xác suất, mỗi model 1 view như lúc val của nó)
    if args.ensemble:
        members = [(model, norm, m0, main_desc)]
        for d in args.ensemble:
            md, mt = load_run_model(Path(d))
            r_, c_ = model_res(mt)
            members.append((md, Normalizer(mt["mean"], mt["std"], "cuda"), get_method("I00", r_, c_),
                            f"{mt['config']['exp_id']} ({mt['weight_tag']})"))
        member_probs = [saved["I00"][1]] + [predict(md, nm, loader, mm)[3] for md, nm, mm, _ in members[1:]]
        for k in range(2, len(members) + 1):
            p = ensemble_probs(member_probs[:k])
            lat = latency_cols([(md, nm, mm) for md, nm, mm, _ in members[:k]], args.iters)
            desc = " + ".join(d for *_, d in members[:k])
            rows.append(metrics_row(f"I05_{k}", f"ensemble {k} model (TB xác suất, 1 view mỗi model)", p, y, k,
                                    desc, lat))
            add_latency_row(f"I05_{k}", desc, k, lat, res)
            print(f"I05_{k}: F1 {rows[-1]['val_macro_f1']:.4f} | p50 amp {lat['p50_ms']:.2f} ms", flush=True)
        # ensemble x TTA D4 (cấu hình ngoại tuyến nặng nhất)
        d4 = [(md, nm, get_method("I09", *model_res({"config": {"img_size": mm.res, "eval_crop_pct": mm.crop_pct}})))
              for md, nm, mm, _ in members]
        pd4 = [saved["I09"][1]] + [predict(md, nm, loader, mm)[3] for md, nm, mm in d4[1:]]
        lat = latency_cols(d4, max(50, args.iters // 2))
        code = f"I05_{len(members)}_D4"
        rows.append(metrics_row(code, f"ensemble {len(members)} model x TTA D4", ensemble_probs(pd4), y,
                                8 * len(members), " + ".join(d for *_, d in members), lat))
        add_latency_row(code, "ensemble x D4", 8 * len(members), lat, res)
        print(f"{code}: F1 {rows[-1]['val_macro_f1']:.4f} | p50 amp {lat['p50_ms']:.2f} ms", flush=True)
        del members, d4
        torch.cuda.empty_cache()

    # 6) I06: EMA vs không EMA (cùng lần chạy, cùng epoch tốt nhất)
    if args.ema_run:
        for ck, code, name in (("best.pt", "I06_EMA", "trọng số EMA"), ("best_raw.pt", "I06_RAW", "trọng số thường")):
            md, mt = load_run_model(Path(args.ema_run), ck)
            nm = Normalizer(mt["mean"], mt["std"], "cuda")
            p = predict(md, nm, loader, get_method("I00", *model_res(mt)))[3]
            rows.append(metrics_row(code, f"{name}, 1 view", p, y, 1,
                                    f"{mt['config']['exp_id']} seed{mt['config']['seed']} (cùng epoch)",
                                    {"note": "EMA không tốn thêm chi phí suy luận"}))
            print(f"{code}: F1 {rows[-1]['val_macro_f1']:.4f}", flush=True)
            del md

    # 7) I08: gộp BN (CNN có BatchNorm), FP32
    if args.bn_run:
        md, mt = load_run_model(Path(args.bn_run))
        nm = Normalizer(mt["mean"], mt["std"], "cuda")
        mb = get_method("I00", *model_res(mt))
        # kiểm tra độ chính xác phép gộp ở FP32 thật: tắt TF32 của cuDNN/cuBLAS (mặc định bật trên Ampere,
        # gây sai lệch ~1e-2 ở logit giữa hai cách tính, không phải do phép gộp); độ trễ vẫn đo ở mặc định
        tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        fused = fuse_conv_bn(md, check_input=torch.randn(4, 3, 224, 224, device="cuda"))
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
        desc = f"{mt['config']['exp_id']} ({mt['weight_tag']})"
        for mm, code, name in ((md, "I08_BN_OFF", "chưa gộp BN"), (fused, "I08_BN_FUSED", "đã gộp BN vào conv")):
            p = predict(mm, nm, loader, mb, amp=False)[3]
            lb = {(b, dt): latency_report(mm, b, mb.res, dt, iters=args.iters if b == 1 else max(50, args.iters // 4),
                                          fused_bn=mm is fused, label=f"{code} | {desc}")
                  for b in (1, 32) for dt in ("fp32", "amp", "fp16")}
            rows.append(metrics_row(code, f"{name}, FP32, 1 view", p, y, 1, desc,
                                    {"p50_ms": lb[(1, "fp32")]["p50_ms"], "p95_ms": lb[(1, "fp32")]["p95_ms"],
                                     "p99_ms": lb[(1, "fp32")]["p99_ms"],
                                     "throughput_img_s": lb[(32, "fp32")]["images_per_s"], "latency_dtype": "fp32",
                                     "n_fused": getattr(fused, "n_fused", None),
                                     "fuse_max_abs_err": getattr(fused, "fuse_max_abs_err", None),
                                     "note": "độ trễ chỉ forward"}))
            for (b, dt), r in lb.items():
                lat_rows.append({"config": r["config"], "gpu": gpu, "dtype": dt, "batch": b, "k_views": 1,
                                 "img_size": mb.res, "fused_bn": mm is fused, "p50_ms": r["p50_ms"],
                                 "p95_ms": r["p95_ms"], "p99_ms": r["p99_ms"], "images_per_s": r["images_per_s"],
                                 "torch": torch.__version__, "includes_preprocessing": "không (chỉ forward)"})
            print(f"{code}: F1 {rows[-1]['val_macro_f1']:.4f} | p50 fp32 {rows[-1]['p50_ms']:.2f} ms", flush=True)
        del md, fused

    df = pd.DataFrame(rows)
    base = df.loc[df.exp_id == "I00"].iloc[0]
    df["delta_f1_vs_I00"] = df["val_macro_f1"] - base["val_macro_f1"]
    df["relative_cost_vs_I00"] = df["p50_ms"] / base["p50_ms"]
    df.to_csv(out / "inference_val.csv", index=False)
    pd.DataFrame(lat_rows).to_csv(out / "latency.csv", index=False)
    print(df[["exp_id", "K", "val_macro_f1", "val_top1", "val_ece", "p50_ms", "p95_ms"]].to_string(index=False))

    # đường đánh đổi độ chính xác - độ trễ (AMP, batch 1)
    d = df[df["exp_id"].str.match(r"^I(0\d|10)") & ~df["exp_id"].str.startswith(("I06", "I07", "I08"))]
    d = d.dropna(subset=["p50_ms"])
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    sc = ax.scatter(d["p50_ms"], d["val_macro_f1"], c=np.log2(d["K"].astype(float)), cmap="viridis", s=45)
    for _, r in d.iterrows():
        ax.annotate(r["exp_id"], (r["p50_ms"], r["val_macro_f1"]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="log2 K (số view x số model)")
    ax.set_xscale("log")
    ax.set_xlabel(f"độ trễ p50 batch 1, AMP (ms, thang log) - {gpu}")
    ax.set_ylabel("macro-F1 val")
    ax.set_title(f"Đánh đổi độ chính xác - độ trễ (model chính: {main_desc})", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(figs / "inference_tradeoff.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
