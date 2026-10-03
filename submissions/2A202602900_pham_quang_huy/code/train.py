"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh (từ thư mục gốc repo, nơi có data/ và eval.py):
    python submissions/<bai>/code/train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc.

Các lựa chọn đã ghi rõ trong báo cáo:
  - LR cập nhật theo BƯỚC (iteration): warmup tuyến tính `warmup_epochs` epoch, rồi cosine về 0.
  - Val loss luôn là CE thường (không trọng số, không smoothing) để so được giữa các cấu hình.
  - Mixed precision FP16 (autocast + GradScaler); bộ nhớ contiguous (channels_last chậm ~8x khi train trên
    torch 2.14 + cuDNN 9.24 + RTX 3060 Laptop, đã đo bằng probe; nên tắt).
  - cudnn.benchmark = True (nhanh hơn, không tái lập bit-exact; cùng seed cho kết quả rất gần nhau).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _find_repo_with_eval() -> Path:
    """Tìm thư mục chứa eval.py gốc (biến môi trường LAB_REPO_DIR, hoặc đi ngược lên từ file này)."""
    env = os.environ.get("LAB_REPO_DIR")
    if env and (Path(env) / "eval.py").exists():
        return Path(env)
    for p in [HERE, *HERE.parents]:
        if (p / "eval.py").exists():
            return p
    raise FileNotFoundError("không tìm thấy eval.py của repo gốc; đặt LAB_REPO_DIR")


REPO_DIR = _find_repo_with_eval()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(1, str(REPO_DIR))

from eval import compute_metrics, save_predictions  # noqa: E402  (eval.py gốc, không sửa)


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn, dùng đặt tên ảnh curves/<exp_id>_<desc>.png
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    eval_crop_pct: float = 0.875      # val/test: Resize(img/crop_pct) + CenterCrop(img); 1.0 = ảnh đầy đủ
    aug: str = "basic"                # basic | color | trivial | randaug | geo
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    mix_prob: float = 1.0             # xác suất áp dụng trộn cho một batch
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64              # batch hiệu dụng (sau tích luỹ gradient)
    accum_steps: int = 1              # micro-batch = batch_size / accum_steps khi thiếu VRAM
    optimizer: str = "adamw"          # adamw | sgd
    momentum: float = 0.9
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    grad_clip: float | None = None
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = False       # đo thật: channels_last làm backward chậm ~8x trên máy này (xem báo cáo)
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    save_checkpoint: bool = True
    max_steps_per_epoch: int | None = None   # chỉ để chạy thử nhanh
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    stem = f"{cfg.exp_id}_{cfg.desc}" if cfg.desc else cfg.exp_id
    if cfg.seed != 0:
        stem += f"_seed{cfg.seed}"
    return Path(cfg.curves_dir) / f"{stem}.png"


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU + CUDA). Worker của DataLoader được seed qua generator
    (dataset.make_loader). cudnn.benchmark = True: nhanh hơn nhưng không bit-exact giữa các lần chạy."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD + momentum, trục E) với 3 nhóm tham số (xem model.param_groups)."""
    import torch
    from model import param_groups
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=cfg.momentum, nesterov=True)
    raise ValueError(f"optimizer không hợp lệ: {cfg.optimizer}")


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật theo BƯỚC (slide trang 55).

    Hệ số nhân LR (LambdaLR giữ nguyên tỉ lệ backbone/head):
        step < W : (step + 1) / W
        step >= W: 0.5 * (1 + cos(pi * (step - W) / (S - W)))
    """
    import torch
    total = max(1, cfg.epochs * steps_per_epoch)
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))

    def factor(step: int) -> float:
        if warm > 0 and step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    - Bản sao riêng `self.module` (deepcopy) dùng để đánh giá; không ảnh hưởng model đang train.
    - Mọi phần tử float của state_dict được làm trơn, kể cả buffer BatchNorm (running_mean/var), giống
      timm ModelEmaV2; buffer nguyên (num_batches_tracked) được chép thẳng.
    - Decay có khởi động: d_t = min(decay, (1 + t) / (10 + t)) để EMA không bị kéo về trọng số ban đầu
      khi số bước ít (~2k bước cho 12 epoch).
    """

    def __init__(self, model, decay: float):
        import torch
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.updates = 0
        self._torch = torch

    def update(self, model) -> None:
        torch = self._torch
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                src = msd[k].detach()
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(src, alpha=1.0 - d)
                else:
                    v.copy_(src)


def _autocast(cfg_amp: bool):
    import torch
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg_amp and torch.cuda.is_available())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, lr_trace: list | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc", "lr", "imgs_per_s", "steps"}.

    - set_train_mode: model.train(), riêng backbone đóng băng thì để eval (BN không cập nhật).
    - Mixup/CutMix (theo xác suất mix_prob) -> mixed_loss; train_acc chỉ tính trên batch KHÔNG trộn.
    - AMP (autocast + GradScaler), tích luỹ gradient `accum_steps`, clip gradient tuỳ chọn.
    - scheduler.step() và ema.update() sau mỗi bước tối ưu.
    """
    import torch
    from losses import mix_batch, mixed_loss
    from model import set_train_mode

    set_train_mode(model)
    mem_fmt = torch.channels_last if cfg.channels_last else torch.contiguous_format
    loss_sum, n_seen, correct, n_clean = 0.0, 0, 0, 0
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    steps = 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=mem_fmt)
        y = y.to(device, non_blocking=True)
        mixed = cfg.mix is not None and np.random.rand() < cfg.mix_prob
        if mixed:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with _autocast(cfg.amp):
            logits = model(x)
            loss = mixed_loss(criterion, logits, targets) if mixed else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss = {loss.item()} (không hữu hạn)")
        scaler.scale(loss / cfg.accum_steps).backward()
        micro += 1
        loss_sum += loss.item() * x.size(0)
        n_seen += x.size(0)
        if not mixed:
            correct += (logits.argmax(1) == y).sum().item()
            n_clean += x.size(0)
        if micro % cfg.accum_steps == 0:
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if lr_trace is not None:
                lr_trace.append([g["lr"] for g in optimizer.param_groups])
            scheduler.step()
            if ema is not None:
                ema.update(model)
            steps += 1
            if cfg.max_steps_per_epoch and steps >= cfg.max_steps_per_epoch:
                break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"train_loss": loss_sum / max(1, n_seen),
            "train_acc": correct / n_clean if n_clean else float("nan"),
            "lr": optimizer.param_groups[0]["lr"], "imgs_per_s": n_seen / dt, "steps": steps,
            "train_time_s": dt}


def evaluate(model, loader, criterion, device, amp: bool = True, channels_last: bool = False):
    """Chạy model trên loader ở chế độ eval, KHÔNG gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9] float32, loss: float).
    Giữ đúng thứ tự của loader (shuffle=False).
    """
    import torch
    model.eval()
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format
    names, ys, outs = [], [], []
    loss_sum, n = 0.0, 0
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=mem_fmt)
            y = y.to(device, non_blocking=True)
            with _autocast(amp):
                logits = model(x)
            logits = logits.float()
            if criterion is not None:
                loss_sum += criterion(logits, y).item() * x.size(0)
            n += x.size(0)
            names.extend(f)
            ys.append(y.cpu().numpy())
            outs.append(logits.cpu().numpy())
    return names, np.concatenate(ys), np.concatenate(outs).astype(np.float32), loss_sum / max(1, n)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_trace: list | None = None) -> None:
    """Đường cong training -> curves/<exp_id>_<mota>.png: loss train/val, macro-F1/top-1 val
    (và acc train trên batch không trộn), LR theo bước (thấy warmup + cosine)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    ncol = 3 if lr_trace else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.0))
    ax = axes[0]
    ax.plot(h["epoch"], h["train_loss"], "o-", label="train loss")
    ax.plot(h["epoch"], h["val_loss"], "s-", label="val loss (CE)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(h["epoch"], h["val_macro_f1"], "o-", label="val macro-F1")
    ax.plot(h["epoch"], h["val_top1"], "s--", label="val top-1")
    if "train_acc" in h and h["train_acc"].notna().any():
        ax.plot(h["epoch"], h["train_acc"], "^:", label="train acc (batch không trộn)")
    if "val_macro_f1_raw" in h and h["val_macro_f1_raw"].notna().any():
        ax.plot(h["epoch"], h["val_macro_f1_raw"], "x-.", label="val macro-F1 (không EMA)")
    best = h.loc[h["val_macro_f1"].idxmax()]
    ax.axvline(best["epoch"], color="gray", ls=":", lw=1)
    ax.annotate(f"best ep {int(best['epoch'])}\nF1 {best['val_macro_f1']:.4f}",
                (best["epoch"], best["val_macro_f1"]), textcoords="offset points", xytext=(-60, -30),
                fontsize=8, arrowprops={"arrowstyle": "->", "lw": 0.6})
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.set_title("Metric")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    if lr_trace:
        lr = np.asarray(lr_trace)
        ax = axes[2]
        steps = np.arange(1, len(lr) + 1)
        ax.plot(steps, lr[:, 0], label="LR nhóm 1 (backbone)")
        if lr.shape[1] > 1:
            ax.plot(steps, lr[:, -1], label="LR nhóm cuối (head)")
        ax.set_yscale("log")
        ax.set_xlabel("bước (iteration)")
        ax.set_ylabel("learning rate")
        ax.set_title("LR theo bước")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def env_info() -> dict:
    import torch
    import torchvision
    import timm
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "timm": timm.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "os": platform.platform()}


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    1. set_seed; tạo run_dir; ghi config.json (+ phiên bản thư viện, GPU)
    2. load_split + check_split (dừng nếu vi phạm S1-S6)
    3. train/val loader (test loader chỉ tạo khi cfg.save_test_predictions)
    4. model, criterion, optimizer, scheduler, scaler, EMA
    5. mỗi epoch: train -> val -> history; checkpoint tốt nhất theo MACRO-F1 VAL (hoà: epoch sớm hơn)
    6. nạp checkpoint tốt nhất, lưu val logits + val_pred.csv
    7. nếu cfg.save_test_predictions (Bước 4): test đúng MỘT lần, lưu logits + predictions/<id>_seed<k>_test.csv
    8. history.csv, ảnh curves, summary.json
    KHÔNG dùng test cho bất kỳ quyết định nào (README.md, S4).
    """
    import torch
    from dataset import (build_transforms, check_split, load_split, make_loader, NUM_CLASSES,
                         PreloadedEvalLoader)
    from losses import build_criterion
    from model import build_model, count_gmacs, count_params, data_config, weight_tag

    t_start = time.perf_counter()
    set_seed(cfg.seed)
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Windows/WDDM: thiếu VRAM thì driver lặng lẽ tràn sang RAM hệ thống (rất chậm) thay vì báo OOM.
        # Giới hạn allocator để lỗi lộ ra sớm (khi đó tăng accum_steps).
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_MEM_FRACTION", "0.92")))
    if cfg.batch_size % cfg.accum_steps:
        raise ValueError("batch_size phải chia hết cho accum_steps")

    # 2. dữ liệu
    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    split_report = check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)

    # 4a. model (tạo trước để biết mean/std của bộ trọng số)
    model = build_model(cfg.backbone, pretrained=True, num_classes=NUM_CLASSES, drop_rate=cfg.drop_rate,
                        init=cfg.init, drop_path_rate=cfg.drop_path_rate)
    dc = data_config(model)
    n_params = count_params(model)
    try:
        gmacs = count_gmacs(model, cfg.img_size)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] không đếm được GMAC: {e}")
        gmacs = float("nan")
    tag = weight_tag(model)

    meta = {"config": dataclasses.asdict(cfg), "env": env_info(), "weight_tag": tag,
            "mean": dc["mean"], "std": dc["std"], "params_M": n_params, "gmacs": gmacs,
            "split_check": {k: split_report[k] for k in ("n", "union", "overlap", "missing_files")}}
    (rd / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. loader
    train_tf = build_transforms(True, cfg.img_size, cfg.aug, dc["mean"], dc["std"])
    eval_tf = build_transforms(False, cfg.img_size, cfg.aug, dc["mean"], dc["std"], crop_pct=cfg.eval_crop_pct)
    micro_bs = cfg.batch_size // cfg.accum_steps
    train_loader = make_loader(train_df, cfg.images_dir, train_tf, micro_bs, train=True,
                               sampler=cfg.sampler, num_workers=cfg.num_workers, seed=cfg.seed)
    # val nạp sẵn vào RAM (không worker): trên Windows mỗi worker là một tiến trình nạp lại torch
    # (~1-1.5 GB bộ nhớ commit); worker val + train cùng lúc từng gây lỗi 1455 "paging file too small".
    # Cùng phép biến đổi với eval_tf (kiểm tra trùng khớp trong tests_code.py).
    val_loader = PreloadedEvalLoader(val_df, cfg.images_dir, cfg.img_size, dc["mean"], dc["std"],
                                     crop_pct=cfg.eval_crop_pct)

    # 4b. tối ưu
    mem_fmt = torch.channels_last if cfg.channels_last else torch.contiguous_format
    model.to(device).to(memory_format=mem_fmt)
    counts = np.bincount(train_df["Label"].to_numpy(), minlength=NUM_CLASSES)   # CHỈ dùng train
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma,
                                counts=counts, beta=cfg.class_weight_beta).to(device)
    val_criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader) // cfg.accum_steps
    if cfg.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    eval_model = ema.module if ema else model

    # 5. vòng epoch
    history, lr_trace = [], []
    best_f1, best_epoch, best_state = -1.0, -1, None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"[{cfg.exp_id} seed{cfg.seed}] {tag} | {n_params:.2f}M params | {gmacs:.2f} GMAC | "
          f"{steps_per_epoch} bước/epoch | micro-batch {micro_bs} x {cfg.accum_steps}", flush=True)
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device,
                             ema=ema, lr_trace=lr_trace)
        t_val = time.perf_counter()
        _, y, logits, vloss = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp, cfg.channels_last)
        m = compute_metrics(y, logits.argmax(1), softmax_np(logits))
        row = {"epoch": epoch, **{k: tr[k] for k in ("train_loss", "train_acc", "lr", "imgs_per_s",
                                                     "train_time_s")},
               "val_loss": vloss, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
               "val_bal_acc": m["balanced_acc"], "val_ece": m["ece"],
               "val_f1_chinee": m["f1"][0], "val_f1_snake": m["f1"][7],
               "val_time_s": time.perf_counter() - t_val}
        if ema is not None:  # để so EMA với trọng số thường (I06)
            _, y2, lg2, vl2 = evaluate(model, val_loader, val_criterion, device, cfg.amp, cfg.channels_last)
            m2 = compute_metrics(y2, lg2.argmax(1), softmax_np(lg2))
            row.update({"val_macro_f1_raw": m2["macro_f1"], "val_top1_raw": m2["top1"], "val_loss_raw": vl2})
        history.append(row)
        improved = m["macro_f1"] > best_f1     # strict: hoà thì giữ epoch sớm hơn
        if improved:
            best_f1, best_epoch = m["macro_f1"], epoch
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in eval_model.state_dict().items()}
            if ema is not None:
                raw_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        print(f"  ep {epoch:2d}/{cfg.epochs} | train loss {tr['train_loss']:.4f} acc {tr['train_acc']:.4f} | "
              f"val loss {vloss:.4f} F1 {m['macro_f1']:.4f} top1 {m['top1']:.4f}"
              + (f" (raw F1 {row['val_macro_f1_raw']:.4f})" if ema else "")
              + f" | {tr['imgs_per_s']:.0f} img/s, {tr['train_time_s']:.0f}s" + (" *" if improved else ""),
              flush=True)
        pd.DataFrame(history).to_csv(rd / "history.csv", index=False)

    peak_mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
    train_time = sum(r["train_time_s"] for r in history)

    # 6. checkpoint tốt nhất -> val
    eval_model.load_state_dict(best_state)
    if cfg.save_checkpoint:
        torch.save({"state_dict": best_state, "epoch": best_epoch, "config": dataclasses.asdict(cfg),
                    "weight_tag": tag}, rd / "best.pt")
        if ema is not None:
            torch.save({"state_dict": raw_state, "epoch": best_epoch, "note": "trọng số thường cùng epoch"},
                       rd / "best_raw.pt")
    names, y, logits, vloss = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp, cfg.channels_last)
    probs = softmax_np(logits)
    m = compute_metrics(y, probs.argmax(1), probs)
    np.save(rd / "val_logits.npy", logits)
    pd.DataFrame({"Filename": names, "y_true": y}).to_csv(rd / "val_files.csv", index=False)
    save_predictions(rd / "val_pred.csv", names, y, probs)

    # 7. test: CHỈ ở Bước 4
    test_summary = None
    if cfg.save_test_predictions:
        test_loader = PreloadedEvalLoader(test_df, cfg.images_dir, cfg.img_size, dc["mean"], dc["std"],
                                          crop_pct=cfg.eval_crop_pct)
        tnames, ty, tlogits, _ = evaluate(eval_model, test_loader, None, device, cfg.amp, cfg.channels_last)
        np.save(rd / "test_logits.npy", tlogits)
        pd.DataFrame({"Filename": tnames, "y_true": ty}).to_csv(rd / "test_files.csv", index=False)
        save_predictions(pred_path(cfg, "test"), tnames, ty, softmax_np(tlogits))
        save_predictions(pred_path(cfg, "val"), names, y, probs)
        test_summary = str(pred_path(cfg, "test"))

    # 8. log + ảnh
    title = f"{cfg.exp_id} · {cfg.backbone}" + (f" · {cfg.desc}" if cfg.desc else "") + f" · seed {cfg.seed}"
    plot_curves(history, curve_path(cfg), title, lr_trace)
    pd.DataFrame(lr_trace).to_csv(rd / "lr_trace.csv", index=False)
    summary = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weight_tag": tag,
               "params_M": n_params, "gmacs": gmacs, "img_size": cfg.img_size, "epochs": cfg.epochs,
               "best_epoch": best_epoch, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
               "val_bal_acc": m["balanced_acc"], "val_ece": m["ece"], "val_nll": m["nll"],
               "val_f1_per_class": m["f1"].tolist(), "val_recall_per_class": m["recall"].tolist(),
               "train_time_per_epoch_s": train_time / len(history), "train_time_total_s": train_time,
               "wall_time_s": time.perf_counter() - t_start, "peak_mem_GiB": peak_mem,
               "train_imgs_per_s": float(np.mean([r["imgs_per_s"] for r in history])),
               "test_predictions": test_summary, "curve": str(curve_path(cfg))}
    (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{cfg.exp_id} seed{cfg.seed}] best ep {best_epoch}: val macro-F1 {m['macro_f1']:.4f} "
          f"top1 {m['top1']:.4f} | {train_time / 60:.1f} phút train | peak {peak_mem:.2f} GiB", flush=True)
    del train_loader, val_loader
    return summary


def _cast(value: str, type_str: str):
    """Ép kiểu chuỗi theo chú thích kiểu (dạng chuỗi vì `from __future__ import annotations`)."""
    parts = [t.strip() for t in type_str.split("|")]
    if value.lower() in ("none", "null") and "None" in parts:
        return None
    base = [t for t in parts if t != "None"][0]
    if base == "bool":
        if value.lower() in ("1", "true", "yes", "y"):
            return True
        if value.lower() in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"không phải bool: {value}")
    if base == "int":
        return int(value)
    if base == "float":
        return float(value)
    return value


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict, ép kiểu theo field của Config."""
    types = {f.name: str(f.type) for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"cần dạng KEY=VALUE, nhận '{pair}'")
        key, value = pair.split("=", 1)
        if key not in types:
            raise KeyError(f"'{key}' không có trong Config (có: {', '.join(types)})")
        out[key] = _cast(value, types[key])
    return out


def main() -> None:
    """`python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
