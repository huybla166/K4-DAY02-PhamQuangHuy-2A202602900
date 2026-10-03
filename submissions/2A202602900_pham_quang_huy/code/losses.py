"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện (giữ nguyên như bộ khung):
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Hàm loss theo `kind`:
      - "ce"          : cross-entropy
      - "ls"          : CE + label smoothing (kw: smoothing, mặc định 0.1)
      - "focal"       : focal loss (kw: gamma, alpha)
      - "ce_weighted" : CE có trọng số lớp (kw: weight = tensor, hoặc counts + beta để tự tính)
    """
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=kw.get("weight"))
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1) or 0.1)
    if kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha"))
    if kind == "ce_weighted":
        weight = kw.get("weight")
        if weight is None:
            weight = class_weights(kw["counts"], kw.get("beta") or 0.0)
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"loss không hợp lệ: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Tự cài đặt: loss = (1 - eps) * NLL(y) + eps * mean_k(-log p_k). Với eps = 0 là đúng CE;
    kết quả trùng nn.CrossEntropyLoss(label_smoothing=eps) (có unit test trong tests_code.py).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        uniform = -logp.mean(dim=1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * uniform).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    Trung bình theo batch. alpha: None hoặc vector trọng số theo lớp (độ dài K).
    gamma = 0 và alpha = None cho đúng cross-entropy (unit test trong tests_code.py).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t).clamp_min(0.0) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: w_c ∝ 1 / n_c, chuẩn hoá về trung bình 1
    - beta > 0: class-balanced (Cui et al. 2019): w_c = (1 - beta) / (1 - beta ** n_c),
      chuẩn hoá tổng trọng số về số lớp (tức trung bình 1)
    """
    n = np.asarray(counts, dtype=np.float64)
    if (n <= 0).any():
        raise ValueError("mỗi lớp phải có ít nhất 1 ảnh")
    if beta and beta > 0:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
    else:
        w = 1.0 / n
    w = w / w.sum() * len(n)
    return torch.tensor(w, dtype=torch.float32)


def _rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator):
    """Hộp CutMix có diện tích mục tiêu (1 - lam) * H * W, tâm ngẫu nhiên, cắt theo biên ảnh."""
    cut_rat = math.sqrt(1.0 - lam)
    ch, cw = int(h * cut_rat), int(w * cut_rat)
    cy, cx = int(rng.integers(h)), int(rng.integers(w))
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix", rng: np.random.Generator | None = None):
    """Trộn một batch ảnh và nhãn. lam ~ Beta(alpha, alpha).

    - mixup : x_mix = lam * x + (1 - lam) * x[perm]
    - cutmix: dán hộp từ x[perm] vào x; lam được tính lại theo DIỆN TÍCH THỰC của hộp sau khi cắt biên
    Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].
    `rng` mặc định là np.random toàn cục (đã seed bởi set_seed).
    """
    rng = rng or np.random.default_rng(np.random.randint(0, 2 ** 31 - 1))
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.as_tensor(rng.permutation(x.size(0)), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = _rand_bbox(h, w, lam, rng)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / float(h * w)
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    Accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
