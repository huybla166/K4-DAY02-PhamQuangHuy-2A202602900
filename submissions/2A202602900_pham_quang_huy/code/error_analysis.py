"""error_analysis.py - phân tích lỗi SAU Bước 4 (không ảnh hưởng tới lựa chọn cấu hình).

1. Ma trận nhầm lẫn (test, cộng qua seed) của chung kết và mốc, đọc từ predictions/.
2. Lưới ảnh test bị đoán sai, đặc biệt cặp Chinee Apple <-> Snake Weed.
3. Grad-CAM (CNN) trên các ảnh bị đoán sai (điểm thưởng).
4. Lệch phân phối tự tạo trên VAL (tối, nhiễu, mờ, tương phản thấp): macro-F1, ECE trước/sau TS (T khớp
   trên val sạch); và thích ứng lúc kiểm tra bằng chuẩn hoá lại thống kê BN cho một CNN có BN
   (thích ứng trên nửa A của val đã biến đổi, đánh giá trên nửa B; không dùng test).

python submissions/<bai>/code/error_analysis.py --final "submissions/<bai>/predictions/F01_seed*_test.csv" \
    --baseline "submissions/<bai>/predictions/T00_seed*_test.csv" --final-run runs/F01/seed0 \
    --method I09 --bn-run runs/B01/seed0
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train  # noqa: E402,F401
from dataset import CLASS_NAMES, load_split  # noqa: E402
from eval import compute_metrics, read_pred  # noqa: E402
from inference import apply_temperature, fit_temperature  # noqa: E402
from inference_exps import model_res  # noqa: E402
from methods import Normalizer, base_view, combined_logits, get_method, load_run_model, raw_loader, run_views  # noqa: E402

SUB = HERE.parent
SHORT = ["Chinee", "Lantana", "Parkins.", "Parthen.", "P.acacia", "Rubber", "Siam", "Snake", "Negative"]


def plot_confusions(groups: dict, path: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(groups), figsize=(7.2 * len(groups), 6.4))
    axes = np.atleast_1d(axes)
    for ax, (title, files) in zip(axes, groups.items()):
        preds = [read_pred(f) for f in files]
        cm = sum(compute_metrics(p.y_true, p.y_pred, p.probs)["confusion"] for p in preds)
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(9):
            for j in range(9):
                if cm[i, j]:
                    ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.1f}%", ha="center", va="center", fontsize=6.5,
                            color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(9), SHORT, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(9), SHORT, fontsize=8)
        ax.set_xlabel("nhãn dự đoán")
        ax.set_ylabel("nhãn thật")
        ax.set_title(f"{title}\n(test, cộng {len(preds)} seed; % theo hàng = recall)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def misclassified_grid(pred_file: str, images_dir: str, path: Path, max_n: int = 24):
    import matplotlib.pyplot as plt
    from PIL import Image
    p = read_pred(pred_file)
    wrong = np.where(p.y_true != p.y_pred)[0]
    hard = [i for i in wrong if {p.y_true[i], p.y_pred[i]} == {0, 7}]
    rest = [i for i in wrong if i not in hard]
    order = hard + sorted(rest, key=lambda i: -p.probs[i].max())
    sel = order[:max_n]
    cols = 6
    rows = max(1, int(np.ceil(len(sel) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), sel):
        with Image.open(Path(images_dir) / p.filenames[i]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"thật: {SHORT[p.y_true[i]]}\nđoán: {SHORT[p.y_pred[i]]} ({p.probs[i].max():.2f})",
                     fontsize=7, color="red" if i in hard else "black")
    fig.suptitle(f"Ảnh test bị đoán sai ({Path(pred_file).name}); đỏ = cặp Chinee Apple <-> Snake Weed "
                 f"({len(hard)}/{len(wrong)} lỗi)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"n_wrong": int(len(wrong)), "n_chinee_snake": int(len(hard)),
            "wrong_files": [str(p.filenames[i]) for i in sel]}


def gradcam(model, norm, x, target: int):
    """Grad-CAM trên bản đồ đặc trưng cuối (forward_features) của CNN timm."""
    model.zero_grad(set_to_none=True)
    feat = model.forward_features(norm(x))
    nhwc = feat.ndim == 4 and feat.shape[-1] == model.num_features and feat.shape[1] != model.num_features
    if nhwc:
        feat = feat.permute(0, 3, 1, 2)
    feat.retain_grad()
    logits = model.forward_head(feat.permute(0, 2, 3, 1) if nhwc else feat)
    logits[0, target].backward()
    w = feat.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feat).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), logits.softmax(1)[0].detach().cpu().numpy()


def gradcam_figure(run_dir: Path, files: list[str], labels: dict, images_dir: str, path: Path):
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    model, meta = load_run_model(run_dir)
    model = model.float().eval()
    norm = Normalizer(meta["mean"], meta["std"], "cuda")
    m1 = get_method("I00", *model_res(meta))
    n = len(files)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.4 * n))
    axes = np.atleast_2d(axes)
    for r, f in enumerate(files):
        with Image.open(Path(images_dir) / f) as im:
            img = im.convert("RGB")
        x = base_view(to_tensor(img).unsqueeze(0).cuda(), m1.res, m1.crop_pct)
        with torch.enable_grad():
            probs = None
            cams = []
            for tgt in (None, labels[f]):
                if probs is None:
                    with torch.no_grad():
                        probs = model(norm(x)).softmax(1)[0].cpu().numpy()
                    tgt = int(probs.argmax())
                cams.append((tgt, gradcam(model, norm, x, tgt)[0]))
        base = x[0].permute(1, 2, 0).cpu().numpy()
        axes[r, 0].imshow(base)
        axes[r, 0].set_title(f"thật: {SHORT[labels[f]]}", fontsize=7)
        for c, (tgt, cam) in enumerate(cams, start=1):
            axes[r, c].imshow(base)
            axes[r, c].imshow(cam, cmap="jet", alpha=0.45)
            kind = "lớp dự đoán" if c == 1 else "lớp thật"
            axes[r, c].set_title(f"Grad-CAM {kind}: {SHORT[tgt]} (p={probs[tgt]:.2f})", fontsize=7)
        for a in axes[r]:
            a.axis("off")
    fig.suptitle(f"Grad-CAM, {meta['config']['exp_id']} seed{meta['config']['seed']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------- lệch phân phối tự tạo ------------------------------------------
def corrupt(x, kind: str, gen: torch.Generator):
    if kind == "clean":
        return x
    if kind == "dark":            # thiếu sáng: giảm 60% độ sáng + gamma
        return (x * 0.4).clamp(0, 1) ** 1.2
    if kind == "bright":
        return (x * 1.6).clamp(0, 1)
    if kind == "noise":           # nhiễu Gauss sigma = 0.08
        return (x + torch.randn(x.shape, generator=gen, device="cpu").to(x.device) * 0.08).clamp(0, 1)
    if kind == "blur":            # mờ Gauss kernel 9, sigma 2.5 (rung/lệch nét)
        from torchvision.transforms.functional import gaussian_blur
        return gaussian_blur(x, [9, 9], [2.5, 2.5])
    if kind == "low_contrast":
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        return ((x - m) * 0.4 + m).clamp(0, 1)
    raise ValueError(kind)


CORRUPTIONS = ["clean", "dark", "bright", "noise", "blur", "low_contrast"]


def predict_corrupt(model, norm, loader, method, kind, amp=True):
    gen = torch.Generator().manual_seed(0)
    zs, ys = [], []
    model.eval()
    for x, y, _ in loader:
        x = corrupt(x.cuda(non_blocking=True), kind, gen)
        outs = run_views(model, norm, x, method, amp=amp)
        zs.append(np.stack([o.cpu().numpy() for o in outs]))
        ys.append(np.asarray(y))
    per_view = list(np.concatenate(zs, axis=1))
    return per_view, np.concatenate(ys)


def bn_adapt(model, norm, loader, kind, idx_set: set, names_order, method):
    """Chuẩn hoá lại thống kê BN (running mean/var tính lại, momentum tích luỹ) trên ảnh đã biến đổi của
    nửa A val; trọng số giữ nguyên. Trả về bản sao đã thích ứng."""
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None
    m.train()
    gen = torch.Generator().manual_seed(1)
    pos = 0
    with torch.no_grad():
        for x, _, f in loader:
            keep = torch.tensor([names_order[pos + i] in idx_set for i in range(len(f))])
            pos += len(f)
            if keep.sum() < 2:
                continue
            xc = corrupt(x[keep].cuda(), kind, gen)
            with torch.autocast("cuda", dtype=torch.float16):
                m(norm(base_view(xc, method.res, method.crop_pct)).contiguous())
    return m.eval()


def shift_analysis(run_dir: Path, method_code: str, val_df, images_dir, workers, bn_run: Path | None):
    loader = raw_loader(val_df, images_dir, 64, workers)
    model, meta = load_run_model(run_dir)
    method = get_method(method_code, *model_res(meta))
    m1 = get_method("I00", *model_res(meta))
    norm = Normalizer(meta["mean"], meta["std"], "cuda")
    rows = []
    T_clean = None
    for meth in ([m1, method] if method.code != "I00" else [m1]):
        for kind in CORRUPTIONS:
            pv, y = predict_corrupt(model, norm, loader, meth, kind)
            z = combined_logits(pv, meth.space)
            p = apply_temperature(z, 1.0)
            if kind == "clean":
                T_clean = fit_temperature(z, y)
            pc = apply_temperature(z, T_clean)
            a, b = compute_metrics(y, p.argmax(1), p), compute_metrics(y, pc.argmax(1), pc)
            rows.append({"model": f"{meta['config']['exp_id']} seed{meta['config']['seed']}", "method": meth.code,
                         "shift": kind, "val_macro_f1": a["macro_f1"], "val_top1": a["top1"],
                         "ece_before_ts": a["ece"], "ece_after_ts_clean_T": b["ece"], "T_clean": T_clean})
            print(rows[-1], flush=True)
    del model
    torch.cuda.empty_cache()

    # thích ứng BN lúc kiểm tra (CNN có BatchNorm)
    if bn_run is not None:
        model, meta = load_run_model(bn_run)
        norm = Normalizer(meta["mean"], meta["std"], "cuda")
        m1 = get_method("I00", *model_res(meta))
        names = val_df.Filename.tolist()
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(names))
        half_a = {names[i] for i in perm[: len(names) // 2]}
        mask_b = np.array([n not in half_a for n in names])
        for kind in CORRUPTIONS:
            pv, y = predict_corrupt(model, norm, loader, m1, kind)
            p = apply_temperature(pv[0], 1.0)
            ad = bn_adapt(model, norm, loader, kind, half_a, names, m1)
            pv2, _ = predict_corrupt(ad, norm, loader, m1, kind)
            p2 = apply_temperature(pv2[0], 1.0)
            a = compute_metrics(y[mask_b], p[mask_b].argmax(1), p[mask_b])
            b = compute_metrics(y[mask_b], p2[mask_b].argmax(1), p2[mask_b])
            rows.append({"model": f"{meta['config']['exp_id']} seed{meta['config']['seed']}", "method": "I00",
                         "shift": kind, "eval_on": "val nửa B", "val_macro_f1": a["macro_f1"], "val_top1": a["top1"],
                         "ece_before_ts": a["ece"], "bn_adapt_macro_f1": b["macro_f1"], "bn_adapt_top1": b["top1"],
                         "bn_adapt_ece": b["ece"]})
            print(rows[-1], flush=True)
            del ad
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--final-run", required=True)
    ap.add_argument("--method", default="I00")
    ap.add_argument("--bn-run")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-shift", action="store_true")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")

    figs = SUB / "figures"
    out = SUB / "logs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fin, base = sorted(glob.glob(args.final)), sorted(glob.glob(args.baseline))
    plot_confusions({"Chung kết": fin, "Mốc T00 + I00": base}, figs / "confusion_test.png")
    info = misclassified_grid(fin[0], args.images, figs / "misclassified_test.png")
    # các cặp nhầm lẫn lớn nhất (cộng seed)
    cm = sum(compute_metrics(p.y_true, p.y_pred, p.probs)["confusion"] for p in map(read_pred, fin))
    pairs = sorted(((int(cm[i, j]), CLASS_NAMES[i], CLASS_NAMES[j]) for i in range(9) for j in range(9) if i != j),
                   reverse=True)[:10]
    info["top_confusions_sum_over_seeds"] = pairs
    labels = dict(zip(read_pred(fin[0]).filenames, read_pred(fin[0]).y_true))
    cam_files = info["wrong_files"][:6]
    try:
        gradcam_figure(Path(args.final_run), cam_files, {f: int(labels[f]) for f in cam_files}, args.images,
                       figs / "gradcam_misclassified.png")
        info["gradcam"] = "figures/gradcam_misclassified.png"
    except Exception as e:  # noqa: BLE001
        info["gradcam_error"] = str(e)
        print("Grad-CAM lỗi:", e)
    (out / "error_analysis.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if not args.skip_shift:
        _, val_df, _ = load_split(args.labels, 0)
        df = shift_analysis(Path(args.final_run), args.method, val_df, args.images, args.workers,
                            Path(args.bn_run) if args.bn_run else None)
        df.to_csv(out / "distribution_shift_val.csv", index=False)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
