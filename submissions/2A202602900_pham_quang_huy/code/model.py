"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện (giữ nguyên như bộ khung):
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
Thêm: set_train_mode(model), weight_tag(model), data_config(model).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số thực tế được ghi lại bằng weight_tag(model).
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", drop_path_rate: float = 0.0):
    """Tạo model phân loại 9 lớp bằng timm (head mới khởi tạo ngẫu nhiên).

    `init` (trục A): "scratch" (không tiền huấn luyện) | "frozen" (đóng băng backbone) | "finetune".
    """
    import timm

    name = SUGGESTED_BACKBONES.get(name, name)
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init}")
    use_pretrained = pretrained and init != "scratch"
    kw = {"num_classes": num_classes, "drop_rate": drop_rate}
    if drop_path_rate:
        kw["drop_path_rate"] = drop_path_rate
    local = _local_weights(name) if use_pretrained else None
    if local is not None:   # cùng bộ trọng số HF, chỉ là tải sẵn về đĩa (mạng chậm / Kaggle offline)
        kw["pretrained_cfg_overlay"] = {"file": str(local)}
    model = timm.create_model(name, pretrained=use_pretrained, **kw)
    model.init_mode = init
    model.loaded_pretrained = use_pretrained
    if init == "frozen":
        freeze_backbone(model)
    return model


def _local_weights(name: str):
    """File `<kiến_trúc>.<tag>.safetensors` trong thư mục $TIMM_WEIGHTS_DIR (nếu có), tag mặc định của timm."""
    import os
    from pathlib import Path
    import timm
    d = os.environ.get("TIMM_WEIGHTS_DIR")
    if not d:
        return None
    cfg = timm.models.get_pretrained_cfg(name)
    full = f"{cfg.architecture}.{cfg.tag}" if cfg is not None and cfg.tag else name
    p = Path(d) / f"{full}.safetensors"
    return p if p.exists() else None


def weight_tag(model) -> str:
    """Tên đầy đủ `kiến_trúc.tag` của bộ trọng số timm (vd. resnet50.a1_in1k), hoặc 'random-init'."""
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    arch = cfg.get("architecture", type(model).__name__)
    if not getattr(model, "loaded_pretrained", True):
        return f"{arch} (random-init)"
    tag = cfg.get("tag")
    return f"{arch}.{tag}" if tag else arch


def data_config(model) -> dict:
    """mean/std/interpolation mà bộ trọng số yêu cầu."""
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    return {"mean": tuple(cfg.get("mean", (0.485, 0.456, 0.406))),
            "std": tuple(cfg.get("std", (0.229, 0.224, 0.225))),
            "input_size": tuple(cfg.get("input_size", (3, 224, 224))),
            "crop_pct": cfg.get("crop_pct", 0.875)}


def _head_param_ids(model) -> set[int]:
    head = model.get_classifier()
    return {id(p) for p in head.parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head (model.get_classifier()).

    BatchNorm của backbone đóng băng phải luôn ở chế độ eval (không cập nhật running_mean/var bằng
    thống kê batch của dữ liệu mới). Vì model.train() bật lại train mode cho mọi module, train loop
    gọi set_train_mode(model) thay cho model.train().
    """
    head_ids = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(); nếu backbone đóng băng thì đặt mọi module ngoài head về eval (BN, dropout)."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        head = model.get_classifier()
        head_modules = set(head.modules())
        for m in model.modules():
            if m not in head_modules:
                m.eval()
        head.train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm tham số như slide Day 2, trang 52.

    - backbone có ndim > 1                 : lr_backbone, weight_decay
    - norm/bias của backbone (ndim <= 1)  : lr_backbone, weight_decay = 0
      (cùng nhóm này: các tham số timm khai báo trong model.no_weight_decay(), vd. pos_embed, cls_token)
    - head mới                              : lr_head, weight_decay (bias của head: wd = 0)
    """
    head_ids = _head_param_ids(model)
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    groups = {"backbone_decay": [], "backbone_no_decay": [], "head_decay": [], "head_no_decay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or name in skip or name.split(".")[-1] in skip
        part = "head" if id(p) in head_ids else "backbone"
        groups[f"{part}_{'no_decay' if no_decay else 'decay'}"].append(p)
    out = []
    for key, params in groups.items():
        if not params:
            continue
        out.append({"params": params, "name": key,
                    "lr": lr_head if key.startswith("head") else lr_backbone,
                    "weight_decay": 0.0 if key.endswith("no_decay") else weight_decay})
    return out


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size, đếm bằng fvcore.FlopCountAnalysis.

    fvcore đếm một phép nhân-cộng (MAC) là 1 "flop", nên total()/1e9 chính là GMAC (không phải FLOPs 2x).
    Một số phép (softmax, GELU, add...) fvcore bỏ qua; số có thể lệch vài % so với công cụ khác.
    """
    from fvcore.nn import FlopCountAnalysis

    was_training = model.training
    model.eval()
    p = next(model.parameters())
    x = torch.zeros(1, 3, img_size, img_size, device=p.device, dtype=p.dtype)
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        total = fca.total()
    model.train(was_training)
    return total / 1e9
