"""tests_code.py - kiểm tra tự viết cho các phần dễ sai (RUBRIC mục H).

Chạy: python -m unittest submissions/<bai>/code/tests_code.py -v   (không cần dữ liệu; GPU tuỳ chọn)
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark  # noqa: E402
import inference  # noqa: E402
import losses  # noqa: E402
import model as model_mod  # noqa: E402
import train  # noqa: E402


class TestLosses(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.logits = torch.randn(64, 9, generator=g) * 3
        self.y = torch.randint(0, 9, (64,), generator=g)

    def test_focal_gamma0_equals_ce(self):
        fl = losses.FocalLoss(gamma=0.0)(self.logits, self.y)
        ce = F.cross_entropy(self.logits, self.y)
        self.assertLess(abs(fl.item() - ce.item()), 1e-6)

    def test_focal_downweights_easy_examples(self):
        self.assertLess(losses.FocalLoss(gamma=2.0)(self.logits, self.y).item(),
                        F.cross_entropy(self.logits, self.y).item())

    def test_focal_alpha_matches_weighted_sum(self):
        alpha = torch.linspace(0.5, 1.5, 9)
        fl = losses.FocalLoss(gamma=0.0, alpha=alpha)(self.logits, self.y)
        ref = (F.cross_entropy(self.logits, self.y, reduction="none") * alpha[self.y]).mean()
        self.assertLess(abs(fl.item() - ref.item()), 1e-6)

    def test_label_smoothing(self):
        self.assertLess(abs(losses.LabelSmoothingCE(0.0)(self.logits, self.y).item()
                            - F.cross_entropy(self.logits, self.y).item()), 1e-6)
        for eps in (0.05, 0.1, 0.3):
            ours = losses.LabelSmoothingCE(eps)(self.logits, self.y).item()
            ref = F.cross_entropy(self.logits, self.y, label_smoothing=eps).item()
            self.assertLess(abs(ours - ref), 1e-5, eps)

    def test_class_weights(self):
        counts = np.array([675, 638, 619, 613, 637, 605, 644, 610, 5464])
        w = losses.class_weights(counts, 0.0)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        np.testing.assert_allclose((w * torch.tensor(counts, dtype=torch.float32)).numpy(),
                                   (w[0] * counts[0]).item(), rtol=1e-5)   # w ∝ 1/n
        wb = losses.class_weights(counts, 0.999)
        self.assertAlmostEqual(wb.sum().item(), 9.0, places=4)
        self.assertGreater(wb[0].item(), wb[8].item())

    def test_build_criterion_kinds(self):
        counts = np.array([10, 10, 10, 10, 10, 10, 10, 10, 90])
        for kind in ("ce", "ls", "focal", "ce_weighted"):
            crit = losses.build_criterion(kind, smoothing=0.1, gamma=2.0, counts=counts, beta=None)
            self.assertTrue(torch.isfinite(crit(self.logits, self.y)))


class TestMix(unittest.TestCase):
    def test_cutmix_lambda_is_pasted_area(self):
        x = torch.zeros(16, 3, 32, 32)
        x[:8] = 1.0                                       # nửa batch trắng, nửa đen
        y = torch.arange(16) % 9
        rng = np.random.default_rng(0)
        for _ in range(50):
            xm, (ya, yb, lam) = losses.mix_batch(x, y, alpha=1.0, mode="cutmix", rng=rng)
            self.assertTrue(torch.equal(ya, y))           # y_a = y, y_b = y[perm]
            self.assertEqual(sorted(yb.tolist()), sorted(y.tolist()))
            # với ảnh nhận hộp từ ảnh khác màu, tỉ lệ pixel bị thay đúng bằng 1 - lam (diện tích thực)
            changed = (xm != x).float().mean(dim=(1, 2, 3))
            diff_color = changed > 0
            if diff_color.any():
                np.testing.assert_allclose(changed[diff_color].numpy(), 1.0 - lam, atol=1e-6)
            self.assertGreaterEqual(lam, 0.0)
            self.assertLessEqual(lam, 1.0)

    def test_mixup_convex_and_loss(self):
        x = torch.randn(8, 3, 8, 8)
        y = torch.arange(8)
        xm, (ya, yb, lam) = losses.mix_batch(x, y, alpha=0.4, mode="mixup", rng=np.random.default_rng(1))
        self.assertTrue(0.0 <= lam <= 1.0)
        logits = torch.randn(8, 9)
        ml = losses.mixed_loss(nn.CrossEntropyLoss(), logits, (ya, yb, lam))
        ref = lam * F.cross_entropy(logits, ya) + (1 - lam) * F.cross_entropy(logits, yb)
        self.assertLess(abs(ml.item() - ref.item()), 1e-6)


class TestModelParts(unittest.TestCase):
    def test_param_groups_no_decay_on_norm_bias(self):
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=9)
        groups = model_mod.param_groups(m, 1e-4, 1e-3, 0.05)
        head_ids = {id(p) for p in m.get_classifier().parameters()}
        n_total = sum(p.numel() for p in m.parameters())
        self.assertEqual(sum(p.numel() for g in groups for p in g["params"]), n_total)
        for g in groups:
            for p in g["params"]:
                if p.ndim <= 1:
                    self.assertEqual(g["weight_decay"], 0.0)
                self.assertEqual(g["lr"], 1e-3 if id(p) in head_ids else 1e-4)

    def test_freeze_keeps_bn_eval(self):
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=9)
        model_mod.freeze_backbone(m)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        self.assertTrue(all(n.startswith("fc.") for n in trainable), trainable)
        model_mod.set_train_mode(m)
        self.assertFalse(m.bn1.training)
        self.assertTrue(m.fc.training)
        rm = m.bn1.running_mean.clone()
        m(torch.randn(4, 3, 64, 64))
        self.assertTrue(torch.equal(rm, m.bn1.running_mean))   # BN không cập nhật

    def test_scheduler_warmup_cosine(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([{"params": [p], "lr": 1e-3}, {"params": [nn.Parameter(torch.zeros(1))], "lr": 1e-2}])
        cfg = train.Config(epochs=10, warmup_epochs=1.0)
        sch = train.build_scheduler(opt, cfg, steps_per_epoch=100)
        lrs = []
        for _ in range(1000):
            lrs.append([g["lr"] for g in opt.param_groups])
            opt.step()
            sch.step()
        lrs = np.array(lrs)
        self.assertAlmostEqual(lrs[99, 0], 1e-3, places=9)           # hết warmup = LR đỉnh
        self.assertLess(lrs[0, 0], 2e-5)                              # bắt đầu gần 0
        self.assertLess(lrs[-1, 0], 1e-6)                             # cosine về gần 0
        np.testing.assert_allclose(lrs[:, 1] / lrs[:, 0], 10.0)       # giữ tỉ lệ head/backbone

    def test_ema(self):
        m = nn.Linear(3, 2)
        ema = train.EMA(m, decay=0.5)
        w0 = ema.module.weight.clone()
        with torch.no_grad():
            m.weight.add_(1.0)
        ema.update(m)
        d = min(0.5, 2 / 11)
        torch.testing.assert_close(ema.module.weight, d * w0 + (1 - d) * m.weight)

    def test_parse_overrides(self):
        o = train.parse_overrides(["seed=3", "loss=focal", "ema_decay=none", "amp=false", "mix=cutmix",
                                   "lr_head=2e-3", "sampler=None"])
        self.assertEqual(o, {"seed": 3, "loss": "focal", "ema_decay": None, "amp": False, "mix": "cutmix",
                             "lr_head": 2e-3, "sampler": None})
        with self.assertRaises(KeyError):
            train.parse_overrides(["nope=1"])


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_exact(self):
        for name in ("resnet18", "efficientnet_b0"):
            m = model_mod.build_model(name, pretrained=False, num_classes=9).eval()
            with torch.no_grad():   # BN có thống kê khác mặc định để phép gộp có ý nghĩa
                for bn in m.modules():
                    if isinstance(bn, nn.BatchNorm2d):
                        bn.running_mean.uniform_(-0.5, 0.5)
                        bn.running_var.uniform_(0.5, 2.0)
                        bn.weight.uniform_(0.5, 1.5)
                        bn.bias.uniform_(-0.2, 0.2)
            x = torch.randn(2, 3, 96, 96)
            fused = inference.fuse_conv_bn(m, check_input=x, verbose=False)
            self.assertGreater(fused.n_fused, 10, name)
            n_bn_left = sum(isinstance(b, nn.BatchNorm2d) for b in fused.modules())
            self.assertEqual(n_bn_left, 0, name)
            self.assertLess(fused.fuse_max_abs_err, 1e-3, name)

    def test_temperature_recovers_known_T(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(20000, 9)) * 2.0
        p = inference.softmax_np(z / 2.5)
        y = np.array([rng.choice(9, p=row) for row in p])
        T = inference.fit_temperature(z, y)
        self.assertAlmostEqual(T, 2.5, delta=0.1)
        q = inference.apply_temperature(z, T)
        np.testing.assert_array_equal(q.argmax(1), z.argmax(1))   # accuracy không đổi

    def test_aggregate_and_views(self):
        a, b = np.random.randn(5, 9), np.random.randn(5, 9)
        for space in ("prob", "logit"):
            p = inference.aggregate_views([a, b], space)
            np.testing.assert_allclose(p.sum(1), 1.0)
        np.testing.assert_allclose(inference.aggregate_views([a, a], "prob"), inference.softmax_np(a))
        x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
        self.assertTrue(torch.equal(inference.view_hflip(inference.view_hflip(x)), x))
        crops = inference.views_multicrop(x, 4, flip=True)
        self.assertEqual(len(crops), 10)
        self.assertTrue(all(c.shape == (2, 3, 4, 4) for c in crops))
        self.assertTrue(torch.equal(crops[0], x[..., :4, :4]))
        ms = inference.views_multiscale(x, [8, 12])
        self.assertEqual([v.shape[-1] for v in ms], [8, 12])

    def test_bench(self):
        r = benchmark.bench(lambda: math.sqrt(2.0), warmup=10, iters=50)
        self.assertTrue(r["p50"] <= r["p95"] <= r["p99"])
        with self.assertRaises(ValueError):
            benchmark.bench(lambda: None, iters=10)


def _data_dir():
    for p in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (p / "data" / "images").is_dir() and (p / "data" / "labels" / "val_subset0.csv").exists():
            return p / "data"
    return None


@unittest.skipIf(_data_dir() is None, "cần data/images và data/labels")
class TestDataPipeline(unittest.TestCase):
    """Ba đường suy luận phải cho cùng một tensor đầu vào: transform val lúc train (DataLoader),
    loader nạp sẵn dùng trong train.py, và view I00 trên GPU của methods.py (ảnh 256 -> cắt giữa 224)."""

    def test_eval_paths_identical(self):
        import pandas as pd
        import dataset
        import methods
        d = _data_dir()
        df = pd.read_csv(d / "labels" / "val_subset0.csv").head(12)
        mean, std = dataset.IMAGENET_MEAN, dataset.IMAGENET_STD
        ref_ds = dataset.DeepWeedsDataset(df, d / "images", dataset.build_transforms(False, 224, mean=mean, std=std))
        ref = torch.stack([ref_ds[i][0] for i in range(len(df))])
        pre = dataset.PreloadedEvalLoader(df, d / "images", 224, mean, std, batch_size=5, threads=2)
        got = torch.cat([x for x, _, _ in pre])
        self.assertEqual([f for _, _, fs in pre for f in fs], df.Filename.tolist())
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=0)
        raw = torch.cat([x for x, _, _ in methods.raw_loader(df, d / "images", 5)])
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        with Image.open(d / "images" / df.Filename.iloc[0]) as im:
            torch.testing.assert_close(raw[0], to_tensor(im.convert("RGB")), atol=1e-6, rtol=0)
        view = methods.Normalizer(mean, std, "cpu")(methods.make_views(raw, methods.get_method("I00"))[0])
        torch.testing.assert_close(view, ref, atol=1e-5, rtol=0)

    def test_batched_tta_equals_separate_forwards(self):
        import methods
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=9).eval()
        x = torch.rand(3, 3, 256, 256)
        norm = methods.Normalizer((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), "cpu")
        for code in ("I09", "I02", "I02S"):
            meth = methods.get_method(code)
            got = methods.run_views(m, norm, x, meth, amp=False, max_batch=7)
            with torch.no_grad():
                ref = [m(norm(v)) for v in methods.make_views(x, meth)]
            self.assertEqual(len(got), meth.k)
            for g, r in zip(got, ref):
                torch.testing.assert_close(g, r, atol=1e-4, rtol=1e-4)

    def test_split_checks_fold0(self):
        import dataset
        d = _data_dir()
        rep = dataset.check_split(*dataset.load_split(d / "labels", 0), d / "images", verbose=False)
        self.assertEqual(rep["union"], 17509)
        self.assertEqual(sum(rep["overlap"].values()), 0)
        self.assertEqual(rep["missing_files"], 0)


if __name__ == "__main__":
    unittest.main()
