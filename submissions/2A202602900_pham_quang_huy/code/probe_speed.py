"""probe_speed.py - đo nhanh thông lượng train (AMP, batch 64) và bộ nhớ đỉnh của từng backbone
bằng dữ liệu ngẫu nhiên, để ước lượng ngân sách GPU trước khi chạy thật (GUIDE.md mục 7).

python submissions/<bai>/code/probe_speed.py --out submissions/<bai>/logs/probe_speed.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_model, count_gmacs, count_params, param_groups, weight_tag  # noqa: E402

NAMES = ["resnet50", "resnext50_32x4d", "convnext_tiny", "deit_small_patch16_224",
         "swin_tiny_patch4_window7_224", "efficientnet_b0", "mobilenetv3_large_100"]


def probe(name, micro, accum, img=224, steps=25):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = build_model(name, pretrained=True, num_classes=9).cuda()
    opt = torch.optim.AdamW(param_groups(m, 1e-4, 1e-3, 0.05))
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(micro, 3, img, img, device="cuda")
    y = torch.randint(0, 9, (micro,), device="cuda")
    m.train()
    for i in range(steps + 5):
        if i == 5:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        for _ in range(accum):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(m(x), y) / accum
            scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    ips = steps * micro * accum / (time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    info = {"name": name, "tag": weight_tag(m), "params_M": count_params(m), "micro": micro, "accum": accum,
            "img": img, "train_imgs_per_s": ips, "peak_GiB": peak, "est_epoch_s": 10501 / ips}
    del m, opt
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs="*", default=NAMES)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--mem-fraction", type=float, default=0.85,
                    help="giới hạn bộ nhớ allocator: tránh WDDM tràn sang RAM hệ thống (sysmem fallback) cực chậm")
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM trống {free / 2**30:.2f} / {total / 2**30:.2f} GiB; giới hạn allocator {args.mem_fraction:.2f}", flush=True)
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    res = []
    for n in args.names:
        for micro, accum in ((64, 1), (32, 2), (16, 4)):
            try:
                r = probe(n, micro, accum, args.img, args.steps)
                try:
                    m = build_model(n, pretrained=False, num_classes=9)
                    r["gmacs"] = count_gmacs(m, args.img)
                except Exception as e:  # noqa: BLE001
                    r["gmacs"] = None
                    r["gmacs_err"] = str(e)[:200]
                print(json.dumps(r), flush=True)
                res.append(r)
                break
            except torch.OutOfMemoryError:
                print(f"{n}: OOM ở micro-batch {micro}", flush=True)
                torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
