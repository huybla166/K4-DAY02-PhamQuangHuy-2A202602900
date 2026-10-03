"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views_fn)   -> (filenames, y_true, [logits_view_1, ...])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _autocast(amp: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and torch.cuda.is_available())


def predict_logits(model, loader, device, view=None, amp: bool = True, channels_last: bool = False):
    """Chạy model trên loader, gom logit theo đúng thứ tự file. `view`: hàm biến đổi batch hoặc None."""
    names, ys, logits = predict_views(model, loader, device, (lambda x: [view(x)]) if view else None,
                                      amp=amp, channels_last=channels_last)
    return names, ys, logits[0]


def predict_views(model, loader, device, views_fn=None, amp: bool = True, channels_last: bool = False):
    """Như predict_logits nhưng `views_fn(x) -> list[batch]` sinh K view; trả về list K mảng logit."""
    model.eval()
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format
    names, ys, outs = [], [], None
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            views = views_fn(x) if views_fn else [x]
            if outs is None:
                outs = [[] for _ in views]
            for k, v in enumerate(views):
                with _autocast(amp):
                    out = model(v.contiguous(memory_format=mem_fmt))
                outs[k].append(out.float().cpu().numpy())
            names.extend(f)
            ys.append(np.asarray(y))
    return names, np.concatenate(ys), [np.concatenate(o).astype(np.float32) for o in outs]


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) theo chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[3])


def views_hflip(x):
    """TTA K = 2: ảnh gốc + bản lật ngang."""
    return [x, view_hflip(x)]


def views_d4(x):
    """TTA theo nhóm đối xứng D4 (K = 8): 4 góc xoay 0/90/180/270 độ x có/không lật ngang.

    Hợp lệ với DeepWeeds vì ảnh chụp nhìn xuống mặt đất, không có hướng "trên/dưới" chuẩn
    (bài báo gốc cũng augment xoay ±360 độ). Ảnh vuông nên kích thước không đổi.
    """
    out = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3)) if k else x
        out += [r, view_hflip(r)]
    return out


def views_rot4(x):
    """TTA K = 4: xoay 0/90/180/270 độ (không lật)."""
    return [torch.rot90(x, k, dims=(2, 3)) if k else x for k in range(4)]


def views_flips(x):
    """TTA K = 4: gốc, lật ngang, lật dọc, lật cả hai (= xoay 180)."""
    return [x, view_hflip(x), torch.flip(x, dims=[2]), torch.flip(x, dims=[2, 3])]


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop kích thước `crop` (4 góc + giữa) từ batch ảnh lớn hơn; flip=True thêm 5 bản lật (K = 10)."""
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError(f"crop {crop} lớn hơn ảnh {h}x{w}")
    top, left = (h - crop) // 2, (w - crop) // 2
    boxes = [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), (top, left)]
    views = [x[..., i:i + crop, j:j + crop] for i, j in boxes]
    if flip:
        views += [view_hflip(v) for v in views]
    return views


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes` (bilinear, antialias). Trả về list các batch.

    CNN có global pooling nhận được mọi kích thước. ViT/DeiT/Swin cố định lưới patch/cửa sổ theo
    224 nên không dùng được trực tiếp (cần nội suy pos-embed): chỉ áp dụng cho CNN.
    """
    out = []
    for s in sizes:
        if x.shape[-1] == s and x.shape[-2] == s:
            out.append(x)
        else:
            out.append(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True))
    return out


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy TTA (slide trang 62).
      - "prob" : trung bình softmax của từng view
      - "logit": trung bình logit rồi softmax
    Trả về xác suất (N, 9) đã chuẩn hoá.
    """
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in logits_per_view])  # (K, N, C)
    if space == "prob":
        p = softmax_np(stack).mean(0)
    elif space == "logit":
        p = softmax_np(stack.mean(0))
    else:
        raise ValueError(f"space không hợp lệ: {space}")
    return p / p.sum(1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file). Chi phí = số mô hình."""
    p = np.mean(np.stack([np.asarray(q, dtype=np.float64) for q in list_of_probs]), axis=0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL của softmax(logit / T) (slide trang 69).

    Tìm lưới thô trên log T trong [-3, 3] rồi tinh bằng LBFGS (float64). KHÔNG khớp T trên test.
    """
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)

    def nll(log_t):
        return F.cross_entropy(z / torch.exp(log_t), y)

    grid = torch.linspace(-3, 3, 121, dtype=torch.float64)
    with torch.no_grad():
        losses = torch.stack([nll(g) for g in grid])
    log_t = grid[losses.argmin()].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(log_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t.detach()))


def apply_temperature(logits, T: float):
    """softmax(logits / T)."""
    return softmax_np(np.asarray(logits, dtype=np.float64) / T)


def _fuse_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma * w / sqrt(var + eps);  b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode)
    fused = fused.to(device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        std = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight if bn.affine else torch.ones_like(std)
        beta = bn.bias if bn.affine else torch.zeros_like(std)
        scale = gamma / std
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        fused.bias.copy_(beta + (b - bn.running_mean) * scale)
    return fused


def fuse_conv_bn(model, check_input=None, verbose: bool = True):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75).

    Duyệt mọi module con; với cặp (Conv2d, BatchNorm2d) đăng ký liền kề (đúng thứ tự dữ liệu chảy ở
    ResNet/ResNeXt/EfficientNet/MobileNet của timm) thì thay conv bằng conv đã gộp và BN bằng Identity.
    BatchNormAct2d của timm (BN + activation) được thay bằng phần activation của nó.
    Trả về BẢN SAO đã gộp (model gốc không đổi). Nếu có `check_input`, in sai số lớn nhất trước/sau.
    Kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm): không áp dụng (0 cặp).
    """
    model = model.eval()
    fused_model = copy.deepcopy(model).eval()
    n_fused = 0
    for parent in fused_model.modules():
        keys = list(parent._modules.keys())
        for a, b in zip(keys, keys[1:]):
            conv, bn = parent._modules[a], parent._modules[b]
            if (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)
                    and conv.out_channels == bn.num_features and bn.track_running_stats):
                parent._modules[a] = _fuse_pair(conv, bn)
                act = getattr(bn, "act", None)
                drop = getattr(bn, "drop", None)
                if act is not None:   # timm BatchNormAct2d: giữ activation (và drop, Identity lúc eval)
                    parent._modules[b] = nn.Sequential(drop or nn.Identity(), act)
                else:
                    parent._modules[b] = nn.Identity()
                n_fused += 1
    fused_model.n_fused = n_fused
    fused_model.fuse_max_abs_err = float("nan")
    if check_input is not None:
        with torch.inference_mode():
            ref = model(check_input).float()
            out = fused_model(check_input).float()
        err = (ref - out).abs().max().item()
        fused_model.fuse_max_abs_err = err
        if verbose:
            print(f"[fuse_conv_bn] gộp {n_fused} cặp conv+BN; sai số logit lớn nhất = {err:.3e}")
    return fused_model
