"""experiments.py - danh sách thí nghiệm (exp_id -> thay đổi so với nền) và trình chạy hàng loạt.

Mọi thí nghiệm đi qua MỘT hàm train.run(Config(...)); ở đây chỉ khai báo khác biệt so với nền.

Chạy (từ thư mục gốc repo):
    python submissions/<bai>/code/experiments.py --ids B01 B02
    python submissions/<bai>/code/experiments.py --group B
    python submissions/<bai>/code/experiments.py --ids F01 --seeds 0 1 2 --test   # chỉ ở Bước 4
Bỏ qua lần chạy đã có summary.json (chạy lại được khi phiên bị ngắt).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUB_DIR = HERE.parent                       # submissions/<bai>/
sys.path.insert(0, str(HERE))

# ----------------------------------------------------------------------------------------------
# Bước 1: backbone (công thức nền T00, seed 0). accum_steps chỉ đổi micro-batch khi thiếu VRAM;
# batch hiệu dụng vẫn là 64 cho mọi backbone.
# ----------------------------------------------------------------------------------------------
BACKBONES = {
    "B01": ("resnet50", "resnet50", {}),
    "B02": ("resnext50", "resnext50_32x4d", {}),
    "B03": ("convnext_tiny", "convnext_tiny", {}),
    "B04": ("deit_small", "deit_small_patch16_224", {}),
    "B05": ("swin_tiny", "swin_tiny_patch4_window7_224", {}),
    "B06": ("efficientnet_b0", "efficientnet_b0", {}),
    "B07": ("mobilenetv3", "mobilenetv3_large_100", {}),
    # Thêm sau khi xem B01 (underfit): cùng kiến trúc ResNet-50, bộ trọng số torchvision gốc (công thức
    # tiền huấn luyện cổ điển, CE + SGD) thay vì a1_in1k (BCE + LAMB) -> tách ảnh hưởng của trọng số khỏi kiến trúc.
    "B08": ("resnet50_tv", "resnet50.tv_in1k", {}),
    # Chẩn đoán (KHÁC công thức nền, không dùng để xếp hạng backbone): mọi CNN có BatchNorm underfit
    # (train acc ~0.92 ở epoch 12) -> thử LR gấp 10 (backbone 1e-3, head 1e-2) cho ResNet-50 a1_in1k.
    "B09": ("resnet50_lr10x", "resnet50", {"lr_backbone": 1e-3, "lr_head": 1e-2}),
}
BACKBONE_NOTES = {
    "B08": "chẩn đoán: cùng kiến trúc B01, trọng số torchvision tv_in1k; công thức nền T00",
    "B09": "chẩn đoán, KHÁC nền: LR backbone 1e-3 / head 1e-2 (x10); không dùng để xếp hạng",
}

# Backbone đi tiếp sang Bước 2-4 (chọn theo kết quả Bước 1, xem báo cáo).
ABLATION_BACKBONE = "convnext_tiny"
ABLATION_EXTRA: dict = {}

# ----------------------------------------------------------------------------------------------
# Bước 2: công thức huấn luyện. Mỗi dòng: (mô tả, trục, khác T00 ở điểm nào, overrides)
# ----------------------------------------------------------------------------------------------
TRAINING = {
    "T00": ("baseline", "-", "công thức nền (GUIDE 1.4)", {}),
    # A. khởi tạo
    "T01": ("scratch", "A", "init=scratch (không tiền huấn luyện)", {"init": "scratch"}),
    "T02": ("frozen", "A", "init=frozen (chỉ train head)", {"init": "frozen"}),
    # B. augmentation
    "T03": ("color", "B", "aug=color (ColorJitter)", {"aug": "color"}),
    "T04": ("trivialaug", "B", "aug=trivial (TrivialAugmentWide)", {"aug": "trivial"}),
    "T05": ("geo_d4", "B", "aug=geo (lật dọc + xoay 90°)", {"aug": "geo"}),
    "T06": ("cutmix", "B", "mix=cutmix (alpha 1.0)", {"mix": "cutmix", "mix_alpha": 1.0}),
    "T07": ("mixup", "B", "mix=mixup (alpha 0.2)", {"mix": "mixup", "mix_alpha": 0.2}),
    # C. loss
    "T08": ("label_smoothing", "C", "loss=ls (eps 0.1)", {"loss": "ls", "label_smoothing": 0.1}),
    "T09": ("focal", "C", "loss=focal (gamma 2)", {"loss": "focal", "focal_gamma": 2.0}),
    "T10": ("ce_weighted", "C", "loss=ce_weighted (1/n_c)", {"loss": "ce_weighted"}),
    # D. cân bằng mẫu
    "T11": ("balanced_sampler", "D", "sampler=balanced", {"sampler": "balanced"}),
    # E. LR / optimizer
    "T12": ("same_lr", "E", "lr_head = lr_backbone = 1e-4", {"lr_head": 1e-4}),
    # F. chính quy hoá
    "T13": ("ema", "F", "EMA trọng số (decay 0.999)", {"ema_decay": 0.999}),
    # G. độ phân giải / thời gian
    "T14": ("res256", "G", "độ phân giải gốc 256: train RRC 256, val/test ảnh đầy đủ 256 (không crop)",
            {"img_size": 256, "eval_crop_pct": 1.0}),
    # F. chính quy hoá (tiếp)
    "T15": ("drop_path", "F", "stochastic depth drop_path_rate = 0.1", {"drop_path_rate": 0.1}),
}

# Kết hợp (điền sau khi có kết quả trục đơn, CHỈ dựa trên val, 1 seed). T00 qua 3 seed: 0.9724 ± 0.0011.
# Vượt 2 std: T13 EMA (+0.0037), T07 Mixup (+0.0036). Tốt kế tiếp nhưng trong nhiễu: T05 D4 (+0.0017).
COMBOS: dict = {
    "C01": ("ema_mixup", "B+F", "EMA 0.999 + Mixup alpha 0.2 (T13 + T07)",
            {"ema_decay": 0.999, "mix": "mixup", "mix_alpha": 0.2}),
    "C02": ("ema_mixup_d4", "B+F", "EMA + Mixup + lật dọc/xoay 90° (T13 + T07 + T05)",
            {"ema_decay": 0.999, "mix": "mixup", "mix_alpha": 0.2, "aug": "geo"}),
}

# Chung kết: công thức chốt bằng choose_final.py (macro-F1 val cao nhất trong T13, T07, C01, C02),
# huấn luyện lại với seed 0, 1, 2. Phương pháp suy luận chốt riêng ở Bước 3 (cũng chỉ trên val).
FINAL: dict = {}
_choice = HERE / "final_choice.json"
if _choice.exists():
    _base = json.loads(_choice.read_text(encoding="utf-8"))["base"]
    _desc, _axis, _diff, _ov = {**TRAINING, **COMBOS}[_base]
    FINAL["F01"] = (f"final_{_desc}", "F", f"chung kết = {_base}: {_diff}", _ov)


def make_config(exp_id: str, seed: int, args):
    from train import Config
    common = dict(images_dir=args.images, labels_dir=args.labels, out_dir=args.runs,
                  pred_dir=str(SUB_DIR / "predictions"), curves_dir=str(SUB_DIR / "curves"),
                  num_workers=args.workers, seed=seed, epochs=args.epochs)
    if "_f" in exp_id:   # điểm thưởng nhiều fold: "<exp>_f<k>" = cấu hình <exp> trên fold k (bộ ba file fold k)
        base, fold = exp_id.rsplit("_f", 1)
        cfg = make_config(base, seed, args)
        return dataclasses.replace(cfg, exp_id=exp_id, fold=int(fold))
    if exp_id in BACKBONES:
        desc, name, extra = BACKBONES[exp_id]
        return Config(exp_id=exp_id, desc=desc, backbone=name, **{**common, **extra})
    for table in (TRAINING, COMBOS, FINAL):
        if exp_id in table:
            desc, _axis, _diff, ov = table[exp_id]
            kw = {**common, "backbone": ABLATION_BACKBONE, **ABLATION_EXTRA, **ov}
            return Config(exp_id=exp_id, desc=desc, **kw)
    raise KeyError(exp_id)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def run_one(exp_id: str, seed: int, args) -> dict | None:
    import torch
    from train import run, run_dir

    cfg = make_config(exp_id, seed, args)
    cfg.save_test_predictions = bool(args.test)
    rd = run_dir(cfg)
    if (rd / "summary.json").exists() and not args.force:
        print(f"[skip] {exp_id} seed{seed} đã có {rd / 'summary.json'}")
        return json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    rd.mkdir(parents=True, exist_ok=True)
    log_f = open(rd / "train.log", "a", encoding="utf-8")
    with contextlib.redirect_stdout(Tee(sys.__stdout__, log_f)):
        print(f"===== {exp_id} seed{seed} {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
        print(json.dumps(dataclasses.asdict(cfg), ensure_ascii=False))
        summary = run(cfg)
        if args.latency:
            summary.update(preliminary_latency(cfg))
            (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log_f.close()
    # bản sao log nhỏ vào thư mục bài nộp để truy ngược exp_id -> log
    dst = SUB_DIR / "logs" / exp_id / f"seed{seed}"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "history.csv", "summary.json", "train.log", "lr_trace.csv"):
        if (rd / name).exists():
            shutil.copy2(rd / name, dst / name)
    torch.cuda.empty_cache()
    return summary


def preliminary_latency(cfg) -> dict:
    """Độ trễ sơ bộ batch 1, FP32, 224 (Bước 1): warmup 10, 100 lần đo, có synchronize."""
    import torch
    from benchmark import latency_report
    from model import build_model
    from train import run_dir

    ck = torch.load(run_dir(cfg) / "best.pt", map_location="cpu", weights_only=False)
    m = build_model(cfg.backbone, pretrained=False, num_classes=9, init="finetune")
    m.load_state_dict(ck["state_dict"])
    m = m.cuda().eval()
    r = latency_report(m, 1, cfg.img_size, "fp32", iters=100)
    print(f"  latency batch1 fp32: p50 {r['p50_ms']:.2f} ms, p95 {r['p95_ms']:.2f} ms")
    return {"latency_b1_fp32_p50_ms": r["p50_ms"], "latency_b1_fp32_p95_ms": r["p95_ms"],
            "latency_b1_fp32_p99_ms": r["p99_ms"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--group", choices=["B", "T", "C", "F"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--test", action="store_true", help="CHỈ Bước 4: ghi dự đoán trên test")
    ap.add_argument("--latency", action="store_true", help="đo độ trễ sơ bộ batch 1 sau khi train")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ids = list(args.ids)
    if args.group:
        ids += list({"B": BACKBONES, "T": TRAINING, "C": COMBOS, "F": FINAL}[args.group])
    for exp_id in ids:
        for seed in args.seeds:
            run_one(exp_id, seed, args)


if __name__ == "__main__":
    main()
