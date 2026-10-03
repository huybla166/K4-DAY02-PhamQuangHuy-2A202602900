"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo (mặc định 200), báo cáo p50, p95, p99 (kèm mean)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý (đọc ảnh, resize, normalize, copy H2D): chỉ đo forward của model với
    đầu vào đã nằm trên GPU. Riêng `end_to_end_latency` đo cả pipeline từ file ảnh (ghi rõ trong bảng).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian `fn()` (mili-giây): warmup rồi bỏ, mỗi lần đo sync -> t0 -> fn -> sync -> t1."""
    if iters < 50:
        raise ValueError("cần >= 50 lần đo")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = np.empty(iters)
    for i in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)), "mean": float(times.mean()),
            "std": float(times.std(ddof=1)), "n": iters, "warmup": warmup}


def _prepare(model, dtype: str, device: str, channels_last: bool):
    model = model.eval()
    if dtype == "fp16":
        model = copy.deepcopy(model).half()
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 200, channels_last: bool = False, fused_bn: bool = False,
                   k_views: int = 1, label: str = "") -> dict:
    """Độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size * k_views, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast FP16) | "fp16" (model.half()).
    k_views > 1: TTA K view được xếp chung thành một batch (cách cài TTA thực tế trên GPU).
    Trả về dict ghi thẳng vào sheet `Latency`.
    """
    m = _prepare(model, dtype, device, channels_last)
    in_dtype = torch.float16 if dtype == "fp16" else torch.float32
    x = torch.randn(batch_size * k_views, 3, img_size, img_size, device=device, dtype=in_dtype)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    use_amp = dtype == "amp"

    def fn():
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            m(x)

    sync = torch.cuda.synchronize if device.startswith("cuda") else None
    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    return {"config": label, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
            "dtype": dtype, "batch": batch_size, "k_views": k_views, "img_size": img_size,
            "fused_bn": fused_bn, "channels_last": channels_last,
            "p50_ms": r["p50"], "p95_ms": r["p95"], "p99_ms": r["p99"], "mean_ms": r["mean"],
            "images_per_s": batch_size / (r["p50"] / 1000.0), "n_iters": r["n"], "warmup": r["warmup"],
            "torch": torch.__version__, "includes_preprocessing": False}


def tta_latency(model, k_views: int, batch_size: int = 1, img_size: int = 224, **kw) -> dict:
    """Độ trễ TTA K view (xếp chung batch) so với K lần một lượt chạy (slide trang 63)."""
    single = latency_report(model, batch_size, img_size, k_views=1, **kw)
    tta = latency_report(model, batch_size, img_size, k_views=k_views, **kw)
    tta["k_times_single_p50_ms"] = k_views * single["p50_ms"]
    tta["single_p50_ms"] = single["p50_ms"]
    return tta


def end_to_end_latency(model, image_paths, transform, device: str = "cuda", dtype: str = "fp32",
                       warmup: int = 10, iters: int = 200, channels_last: bool = False) -> dict:
    """Độ trễ batch 1 TÍNH CẢ tiền xử lý: đọc JPEG -> transform -> H2D -> forward -> softmax -> argmax."""
    from PIL import Image

    m = _prepare(model, dtype, device, channels_last)
    paths = list(image_paths)
    state = {"i": 0}
    use_amp = dtype == "amp"

    def fn():
        p = paths[state["i"] % len(paths)]
        state["i"] += 1
        with Image.open(p) as im:
            x = transform(im.convert("RGB")).unsqueeze(0)
        x = x.to(device)
        if dtype == "fp16":
            x = x.half()
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = m(x)
        out.float().softmax(1).argmax(1).item()   # .item() buộc chờ GPU xong

    r = bench(fn, warmup=warmup, iters=iters, sync=torch.cuda.synchronize)
    return {"p50_ms": r["p50"], "p95_ms": r["p95"], "p99_ms": r["p99"], "mean_ms": r["mean"],
            "dtype": dtype, "batch": 1, "includes_preprocessing": True}
