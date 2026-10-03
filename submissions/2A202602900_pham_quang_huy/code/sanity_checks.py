"""sanity_checks.py - kiểm tra pipeline trước khi chạy thật (slide trang 59, GUIDE.md mục 1.3).

  1) cố định seed
  2) loss ban đầu của head mới so với -ln(1/9) = 2.197
  3) overfit một batch nhỏ (16 ảnh) tới loss gần 0
  4) ảnh sau augmentation (giải chuẩn hoá) kèm nhãn; ảnh sau CutMix / Mixup kèm lam
  5) model.eval(): đầu ra của một ảnh không phụ thuộc các ảnh khác trong batch; train() thì có

Chạy: python submissions/<bai>/code/sanity_checks.py --out submissions/<bai> --backbone resnet50
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CLASS_NAMES, DeepWeedsDataset, build_transforms, load_split  # noqa: E402
from losses import mix_batch  # noqa: E402
from model import build_model, data_config, param_groups, weight_tag  # noqa: E402
from train import set_seed  # noqa: E402


def denorm(t, mean, std):
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (t.cpu().float() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbones", nargs="+", default=["resnet50"])
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    set_seed(0)
    train_df, val_df, _ = load_split(args.labels, 0)
    res = {"expected_initial_loss": math.log(9)}

    for bb in args.backbones:
        set_seed(0)
        model = build_model(bb, pretrained=True, num_classes=9).to(dev)
        dc = data_config(model)
        r = {"weight_tag": weight_tag(model)}

        # 2) loss ban đầu trên 512 ảnh val (eval transform), head chưa huấn luyện
        ds = DeepWeedsDataset(val_df.sample(512, random_state=0), args.images,
                              build_transforms(False, 224, mean=dc["mean"], std=dc["std"]))
        dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
        model.eval()
        losses, ents, maxp = [], [], []
        with torch.inference_mode():
            for x, y, _ in dl:
                lg = model(x.to(dev)).float()
                losses.append(F.cross_entropy(lg, y.to(dev), reduction="none").cpu())
                p = lg.softmax(1)
                maxp.append(p.max(1).values.cpu())
        r["initial_loss"] = float(torch.cat(losses).mean())
        r["initial_mean_max_prob"] = float(torch.cat(maxp).mean())
        print(f"[{bb}] loss ban đầu = {r['initial_loss']:.4f} (kỳ vọng ≈ {math.log(9):.4f}), "
              f"max prob trung bình {r['initial_mean_max_prob']:.3f}")

        # 3) overfit 16 ảnh (2 ảnh/lớp, không augmentation), 150 bước
        sub = train_df.groupby("Label").sample(2, random_state=0).head(16)
        ds = DeepWeedsDataset(sub, args.images, build_transforms(False, 224, mean=dc["mean"], std=dc["std"]))
        xb = torch.stack([ds[i][0] for i in range(len(ds))]).to(dev)
        yb = torch.tensor([ds[i][1] for i in range(len(ds))], device=dev)
        model.train()
        opt = torch.optim.AdamW(param_groups(model, 1e-4, 1e-3, 0.0))
        curve = []
        for step in range(150):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            curve.append(loss.item())
        model.eval()
        with torch.inference_mode():
            acc = (model(xb).argmax(1) == yb).float().mean().item()
        r.update({"overfit_first_loss": curve[0], "overfit_final_loss": curve[-1], "overfit_train_acc": acc})
        r["overfit_curve"] = curve
        print(f"[{bb}] overfit 16 ảnh: loss {curve[0]:.3f} -> {curve[-1]:.5f}, acc {acc:.3f}")

        # 5) eval mode: kết quả không phụ thuộc batch; train mode (BN) thì có
        x = xb[:8]
        with torch.inference_mode():
            model.eval()
            a = model(x)[:1].float()
            b = model(torch.cat([x[:1], torch.randn_like(x[1:]) * 3]))[:1].float()
            r["eval_mode_batch_dependence"] = float((a - b).abs().max())
            model.train()
            a = model(x)[:1].float()
            b = model(torch.cat([x[:1], torch.randn_like(x[1:]) * 3]))[:1].float()
            r["train_mode_batch_dependence"] = float((a - b).abs().max())
        model.eval()
        print(f"[{bb}] |Δ logit| khi đổi ảnh khác trong batch: eval {r['eval_mode_batch_dependence']:.2e}, "
              f"train {r['train_mode_batch_dependence']:.2e}")
        res[bb] = r
        del model, opt
        torch.cuda.empty_cache()

    # biểu đồ overfit
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for bb in args.backbones:
        ax.plot(res[bb]["overfit_curve"], label=f"{bb} (cuối {res[bb]['overfit_final_loss']:.4f})")
    ax.axhline(math.log(9), color="gray", ls=":", label="ln 9")
    ax.set_yscale("log")
    ax.set_xlabel("bước")
    ax.set_ylabel("CE loss (16 ảnh)")
    ax.set_title("Sanity check: overfit một batch nhỏ")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "sanity_overfit.png", dpi=130)
    plt.close(fig)

    # 4) ảnh sau augmentation
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    rows = ["basic", "color", "trivial", "geo"]
    files = train_df.sample(6, random_state=1)
    fig, axes = plt.subplots(len(rows) + 1, 6, figsize=(13, 2.3 * (len(rows) + 1)))
    ds0 = DeepWeedsDataset(files, args.images, build_transforms(False, 224, mean=mean, std=std))
    for j in range(6):
        img, lab, _ = ds0[j]
        axes[0, j].imshow(denorm(img, mean, std))
        axes[0, j].set_title(f"gốc (eval)\n{CLASS_NAMES[lab]}", fontsize=8)
    for i, aug in enumerate(rows, start=1):
        torch.manual_seed(i)
        ds = DeepWeedsDataset(files, args.images, build_transforms(True, 224, aug, mean, std))
        for j in range(6):
            img, lab, _ = ds[j]
            axes[i, j].imshow(denorm(img, mean, std))
            axes[i, j].set_title(f"aug={aug}\n{CLASS_NAMES[lab]}", fontsize=8)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Ảnh sau augmentation (đã giải chuẩn hoá), nhãn ghi trên từng ảnh", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "sanity_augmentations.png", dpi=110)
    plt.close(fig)

    # CutMix / Mixup
    ds = DeepWeedsDataset(train_df.sample(8, random_state=2), args.images,
                          build_transforms(False, 224, mean=mean, std=std))
    xb = torch.stack([ds[i][0] for i in range(8)])
    yb = torch.tensor([ds[i][1] for i in range(8)])
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for r_i, mode in enumerate(["cutmix", "mixup"]):
        xm, (ya, yb2, lam) = mix_batch(xb, yb, 1.0, mode, rng=np.random.default_rng(r_i))
        for j in range(4):
            axes[r_i, j].imshow(denorm(xm[j], mean, std))
            axes[r_i, j].set_title(f"{mode} lam={lam:.2f}\n{CLASS_NAMES[ya[j]]} / {CLASS_NAMES[yb2[j]]}",
                                   fontsize=8)
            axes[r_i, j].set_xticks([])
            axes[r_i, j].set_yticks([])
        res[f"{mode}_lam_example"] = lam
    fig.suptitle("CutMix / Mixup: nhãn y_a / y_b và lam (CutMix: lam = 1 - diện tích hộp thực)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "sanity_mix.png", dpi=110)
    plt.close(fig)

    for bb in args.backbones:
        res[bb].pop("overfit_curve")
    (out / "logs" / "sanity.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
