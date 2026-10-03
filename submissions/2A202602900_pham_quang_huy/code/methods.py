"""methods.py - định nghĩa các phương pháp suy luận (I00...) dùng chung cho Bước 3 (val) và Bước 4 (test).

Loader trả ảnh gốc 256x256 dạng [0, 1] (chưa chuẩn hoá); mọi view (crop, lật, xoay, đổi độ phân giải)
và chuẩn hoá theo mean/std của từng model được làm trên GPU. Với I00, phép cắt giữa 224 trên GPU trùng
với transform lúc train/val (Resize(256) là no-op với ảnh 256 + CenterCrop(224)); inference_exps.py kiểm
tra điều này bằng cách so với val_logits.npy đã lưu khi train.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import (aggregate_views, views_d4, views_flips, views_hflip,  # noqa: E402
                       views_multicrop)

NATIVE = 256


@dataclass
class Method:
    code: str
    name: str
    views: str = "single"          # single | hflip | d4 | flips | crop10 | multiscale
    res: int = 224                 # độ phân giải đưa vào model
    crop_pct: float = 0.875        # view gốc: Resize(res / crop_pct) + CenterCrop(res); 1.0 = ảnh đầy đủ
    space: str = "prob"            # gộp view: prob | logit
    scales: tuple = ()
    note: str = ""
    k: int = field(init=False)

    def __post_init__(self):
        self.k = {"single": 1, "hflip": 2, "d4": 8, "flips": 4, "crop10": 10,
                  "multiscale": max(1, len(self.scales))}[self.views]


def center_crop(x, size: int):
    h, w = x.shape[-2:]
    t, l = (h - size) // 2, (w - size) // 2
    return x[..., t:t + size, l:l + size]


def resize(x, size: int):
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


def base_view(x, res: int, crop_pct: float):
    """Giống transform eval lúc train: Resize(round(res / crop_pct)) rồi CenterCrop(res).
    (res 224, crop 0.875: ảnh 256 giữ nguyên rồi cắt giữa 224; res 256, crop 1.0: ảnh đầy đủ 256.)"""
    return center_crop(resize(x, int(round(res / crop_pct))), res)


def make_views(x, m: Method):
    """x: batch [0,1] ở 256x256 -> list các batch view (chưa chuẩn hoá)."""
    if m.views == "multiscale":
        return [resize(x, s) for s in m.scales]
    if m.views == "crop10":   # 5 crop 87.5% cạnh (4 góc + giữa) + bản lật, đưa về res
        crop = int(round(NATIVE * 0.875))
        return [resize(v, m.res) for v in views_multicrop(x, crop, flip=True)]
    base = base_view(x, m.res, m.crop_pct)
    if m.views == "single":
        return [base]
    if m.views == "hflip":
        return views_hflip(base)
    if m.views == "d4":
        return views_d4(base)
    if m.views == "flips":
        return views_flips(base)
    raise ValueError(m.views)


def build_methods(res: int = 224, crop_pct: float = 0.875) -> list[Method]:
    """Danh sách phương pháp suy luận cho một model train ở `res` (view gốc như lúc val)."""
    base = "ảnh 256 cắt giữa 224" if (res, crop_pct) == (224, 0.875) else \
        "ảnh đầy đủ 256" if (res, crop_pct) == (256, 1.0) else f"resize {round(res / crop_pct)} + crop {res}"
    b = {"res": res, "crop_pct": crop_pct}
    out = [
        Method("I00", f"1 view ({base})", "single", **b),
        Method("I01", "TTA lật ngang (K=2), gộp xác suất", "hflip", space="prob", **b),
        Method("I01L", "TTA lật ngang (K=2), gộp logit", "hflip", space="logit", **b),
        Method("I02", "TTA 5 crop 224 + lật (K=10)", "crop10", space="prob", **b),
        Method("I02S", f"TTA đa tỉ lệ ảnh đầy đủ {{{res},{res + 32},{res + 64}}} (K=3)", "multiscale",
               space="prob", scales=(res, res + 32, res + 64), **b),
    ]
    for r in (224, 256, 288, 320):
        out.append(Method(f"I04_{r}", f"1 view, ảnh đầy đủ ở {r} (dò độ phân giải)", "single", res=r, crop_pct=1.0))
    out += [
        Method("I09", "TTA D4: 4 góc xoay x lật (K=8), gộp xác suất", "d4", space="prob", **b),
        Method("I09L", "TTA D4 (K=8), gộp logit", "d4", space="logit", **b),
        Method("I10", "TTA 4 phép lật (K=4)", "flips", space="prob", **b),
    ]
    return out


def get_method(code: str, res: int = 224, crop_pct: float = 0.875) -> Method:
    for m in build_methods(res, crop_pct):
        if m.code == code:
            return m
    raise KeyError(code)


class Normalizer:
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def __call__(self, x):
        return (x - self.mean) / self.std


def run_views(model, norm: Normalizer, x, m: Method, amp: bool = True, dtype=None, max_batch: int = 256):
    """Trả về list logit (float32) của từng view cho batch x (đã ở GPU).

    Các view cùng kích thước được XẾP CHUNG một batch (K x B ảnh, một lượt forward, chia khúc nếu quá
    `max_batch`), đúng cách cài TTA thực tế trên GPU; view khác kích thước (đa tỉ lệ) chạy riêng.
    Model ở eval() nên kết quả từng ảnh không phụ thuộc ảnh khác trong batch (kiểm tra ở sanity check).
    """
    views = make_views(x, m)
    outs = [None] * len(views)
    groups: dict = {}
    for i, v in enumerate(views):
        groups.setdefault(tuple(v.shape[-2:]), []).append(i)
    with torch.inference_mode():
        for idx in groups.values():
            v = norm(torch.cat([views[i] for i in idx])).contiguous()
            if dtype is not None:
                v = v.to(dtype)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp and dtype is None):
                o = torch.cat([model(c) for c in v.split(max_batch)]).float()
            for j, i in enumerate(idx):
                outs[i] = o[j * x.size(0):(j + 1) * x.size(0)]
    return outs


def predict(model, norm, loader, m: Method, device="cuda", amp: bool = True, dtype=None):
    """Chạy phương pháp m trên cả loader. Trả về (names, y, logits_per_view [K x (N,9)], probs (N,9))."""
    model.eval()
    names, ys, per_view = [], [], None
    for x, y, f in loader:
        x = x.to(device, non_blocking=True)
        outs = run_views(model, norm, x, m, amp, dtype)
        if per_view is None:
            per_view = [[] for _ in outs]
        for k, o in enumerate(outs):
            per_view[k].append(o.cpu().numpy())
        names.extend(f)
        ys.append(np.asarray(y))
    per_view = [np.concatenate(v).astype(np.float32) for v in per_view]
    probs = aggregate_views(per_view, m.space)
    return names, np.concatenate(ys), per_view, probs


def combined_logits(per_view, space: str):
    """Một 'logit' đại diện cho K view để khớp temperature: trung bình logit (space=logit) hoặc
    log của xác suất trung bình (space=prob)."""
    if len(per_view) == 1:
        return per_view[0].astype(np.float64)
    if space == "logit":
        return np.mean(np.stack(per_view), 0).astype(np.float64)
    return np.log(np.clip(aggregate_views(per_view, "prob"), 1e-12, None))


def load_run_model(run_dir: Path, ckpt: str = "best.pt", device: str = "cuda"):
    """Dựng lại model của một lần chạy từ config.json + checkpoint (không tải trọng số ImageNet)."""
    from model import build_model
    meta = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    model = build_model(cfg["backbone"], pretrained=False, num_classes=9, init="finetune",
                        drop_rate=cfg.get("drop_rate", 0.0), drop_path_rate=cfg.get("drop_path_rate", 0.0))
    ck = torch.load(Path(run_dir) / ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    return model, meta


def raw_loader(df, images_dir, batch_size=64, num_workers=6):
    """Ảnh gốc 256 dạng [0,1] (= Resize(256) + CenterCrop(256) + ToTensor, đều là no-op hình học với ảnh
    256x256), giữ thứ tự df. Nạp sẵn một lần vào RAM (uint8) để duyệt nhiều lượt mà không spawn worker
    (Windows). `num_workers` giữ cho tương thích chữ ký cũ, không dùng."""
    from dataset import PreloadedEvalLoader
    return PreloadedEvalLoader(df, images_dir, NATIVE, crop_pct=1.0, batch_size=batch_size, normalize=False)
