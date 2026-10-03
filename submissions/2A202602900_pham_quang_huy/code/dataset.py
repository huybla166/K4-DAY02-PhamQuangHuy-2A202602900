"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện (giữ nguyên như bộ khung):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Lựa chọn đã ghi rõ trong báo cáo:
  - Val/test: ảnh gốc 256x256 -> Resize(round(img_size / crop_pct)) -> CenterCrop(img_size), crop_pct = 0.875
    (với img_size = 224: Resize(256) là no-op, rồi CenterCrop(224)).
  - Chuẩn hoá theo mean/std của bộ trọng số timm (ImageNet cho CNN; 0.5/0.5 cho DeiT/ViT augreg...).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 9
TOTAL_IMAGES = 17509
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)
EVAL_CROP_PCT = 0.875


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1), không sửa gì.

    Lưu ý: file *_subset*.csv của tác giả chỉ có cột `Filename, Label` (cột `Species` chỉ có trong labels.csv).
    """
    labels_dir = Path(labels_dir)
    out = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{split}_subset{fold}.csv thiếu cột {missing}")
        out.append(df)
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True) -> dict:
    """Các kiểm tra bắt buộc trước khi train (README.md, mục 2.1). Sai thì raise để dừng ngay."""
    images_dir = Path(images_dir)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: int(len(v)) for k, v in splits.items()}
    total = sum(n.values())
    frac = {k: v / total for k, v in n.items()}
    per_class = {k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).astype(int).tolist()
                 for k, v in splits.items()}

    # 1. không trùng tên file trong cùng một tập, và giao từng cặp tập phải rỗng
    for k, v in splits.items():
        dup = int(v["Filename"].duplicated().sum())
        if dup:
            raise AssertionError(f"{k}: có {dup} Filename bị trùng")
    names = {k: set(v["Filename"]) for k, v in splits.items()}
    overlap = {"train∩val": len(names["train"] & names["val"]),
               "train∩test": len(names["train"] & names["test"]),
               "val∩test": len(names["val"] & names["test"])}
    if any(overlap.values()):
        raise AssertionError(f"giao giữa các tập khác rỗng: {overlap}")

    # 2. hợp ba tập đúng 17.509 ảnh
    union = names["train"] | names["val"] | names["test"]
    if len(union) != TOTAL_IMAGES:
        raise AssertionError(f"hợp ba tập có {len(union)} ảnh, kỳ vọng {TOTAL_IMAGES}")

    # 3. tỉ lệ xấp xỉ 60/20/20 (lệch > 1 điểm phần trăm thì dừng)
    for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        if abs(frac[k] - target) > 0.01:
            raise AssertionError(f"{k}: tỉ lệ {frac[k]:.4f} lệch quá 1 điểm % so với {target}")

    # 4. mọi file tồn tại
    on_disk = {p.name for p in images_dir.iterdir()} if images_dir.exists() else set()
    missing = sorted(union - on_disk)
    if missing:
        raise AssertionError(f"{len(missing)} file trong CSV không có trong {images_dir}, ví dụ {missing[:3]}")

    # nhãn hợp lệ, và mỗi Filename chỉ có một nhãn
    for k, v in splits.items():
        if not v["Label"].between(0, NUM_CLASSES - 1).all():
            raise AssertionError(f"{k}: Label ngoài 0..{NUM_CLASSES - 1}")

    report = {"n": n, "fraction": frac, "total": total, "union": len(union), "overlap": overlap,
              "per_class": per_class, "class_names": CLASS_NAMES, "missing_files": len(missing),
              "images_on_disk": len(on_disk)}
    if verbose:
        print(f"[check_split] n = {n} (tỉ lệ {', '.join(f'{k} {v:.3f}' for k, v in frac.items())}); "
              f"hợp = {len(union)}; giao = {overlap}; thiếu file = {len(missing)}")
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_pct: float = EVAL_CROP_PCT):
    """Tạo transform (torchvision.transforms v1).

    `aug` (trục B của GUIDE.md mục 3):
      - "basic"  : RandomResizedCrop(img_size) + lật ngang
      - "color"  : basic + ColorJitter(0.3, 0.3, 0.3, 0.05)
      - "trivial": basic + TrivialAugmentWide
      - "randaug": basic + RandAugment(num_ops=2, magnitude=9)
      - "geo"    : basic + lật dọc + xoay bội số 90 độ (ảnh cỏ chụp nhìn xuống, không có "trên/dưới"
                   chuẩn; bài báo gốc xoay ±360 độ)
    Mixup/CutMix trộn theo batch nên nằm ở losses.py.

    Val/test: Resize(round(img_size / crop_pct)) + CenterCrop(img_size), không ngẫu nhiên.
    """
    from torchvision import transforms as T

    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        resize = int(round(img_size / crop_pct))
        return T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
                          T.CenterCrop(img_size), *norm])

    base = [T.RandomResizedCrop(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip()]
    if aug == "basic":
        extra = []
    elif aug == "color":
        extra = [T.ColorJitter(0.3, 0.3, 0.3, 0.05)]
    elif aug == "trivial":
        extra = [T.TrivialAugmentWide(interpolation=T.InterpolationMode.BILINEAR)]
    elif aug == "randaug":
        extra = [T.RandAugment(num_ops=2, magnitude=9, interpolation=T.InterpolationMode.BILINEAR)]
    elif aug == "geo":
        extra = [T.RandomVerticalFlip(), RandomRot90()]
    else:
        raise ValueError(f"aug không hợp lệ: {aug}")
    return T.Compose([*base, *extra, *norm])


class RandomRot90:
    """Xoay ảnh PIL một góc ngẫu nhiên trong {0, 90, 180, 270} độ (không nội suy, không mất góc)."""

    def __call__(self, img):
        k = random.randint(0, 3)
        if k == 0:
            return img
        from PIL import Image
        return img.transpose([Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                              Image.Transpose.ROTATE_270][k - 1])

    def __repr__(self):
        return "RandomRot90()"


def _torch_dataset_base():
    import torch.utils.data
    return torch.utils.data.Dataset


class DeepWeedsDataset(_torch_dataset_base()):
    """Đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) -> (ảnh đã transform, nhãn int, tên file str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.filenames = df["Filename"].astype(str).tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def load_image(self, i: int):
        from PIL import Image
        with Image.open(self.images_dir / self.filenames[i]) as im:
            return im.convert("RGB")

    def __getitem__(self, i: int):
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.filenames[i]


class PreloadedEvalLoader:
    """Loader đánh giá KHÔNG dùng tiến trình worker: giải mã + Resize/CenterCrop toàn bộ ảnh MỘT lần
    (uint8 trong RAM, nhiều luồng), mỗi batch chỉ còn ToTensor (/255) + Normalize.

    Cho kết quả trùng make_loader(build_transforms(False, ...), train=False): cùng phép biến đổi, cùng
    thứ tự df. Lý do: trên Windows mỗi worker là một tiến trình nạp lại torch (~1-1.5 GB bộ nhớ commit);
    worker val + worker train cùng lúc từng gây lỗi 1455 "paging file too small" (lần chạy B03 đầu tiên).
    Phát ra (x float32 đã chuẩn hoá, y int64, filenames) như DataLoader.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, img_size: int = 224,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_pct: float = EVAL_CROP_PCT,
                 batch_size: int = 128, threads: int = 8, normalize: bool = True):
        from concurrent.futures import ThreadPoolExecutor

        import torch
        from torchvision import transforms as T

        geo = T.Compose([T.Resize(int(round(img_size / crop_pct)), interpolation=T.InterpolationMode.BICUBIC),
                         T.CenterCrop(img_size)])
        ds = DeepWeedsDataset(df, images_dir, None)

        def load(i):
            return np.asarray(geo(ds.load_image(i)), dtype=np.uint8)

        with ThreadPoolExecutor(threads) as ex:
            imgs = list(ex.map(load, range(len(ds))))
        self.x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()   # N,3,H,W uint8
        self.y = torch.as_tensor(ds.labels, dtype=torch.long)
        self.filenames = ds.filenames
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.batch_size = batch_size
        self.normalize = normalize

    def __len__(self) -> int:
        return (len(self.y) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for s in range(0, len(self.y), self.batch_size):
            x = self.x[s:s + self.batch_size].float().div_(255)
            if self.normalize:
                x = x.sub_(self.mean).div_(self.std)
            yield x, self.y[s:s + self.batch_size], self.filenames[s:s + self.batch_size]


def seed_worker(worker_id: int) -> None:
    """Seed cho worker của DataLoader: lấy từ torch.initial_seed() (đã do generator quyết định)."""
    import torch
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2, seed: int = 0,
                persistent_workers: bool | None = None):
    """Tạo DataLoader.

    - train=True: shuffle (hoặc sampler "balanced" = WeightedRandomSampler, trọng số 1/n_c, có hoàn lại,
      số mẫu mỗi epoch = len(df)); drop_last=True để tránh batch cuối quá nhỏ làm BN không ổn định.
    - train=False: giữ nguyên thứ tự df (để ghép logit với Filename), không drop.
    - generator + worker_init_fn để tái lập thứ tự batch và augmentation theo seed.
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = DeepWeedsDataset(df, images_dir, transform)
    g = torch.Generator()
    g.manual_seed(seed)
    smp, shuffle = None, False
    if train:
        if sampler == "balanced":
            labels = df["Label"].astype(int).to_numpy()
            counts = np.bincount(labels, minlength=NUM_CLASSES)
            w = 1.0 / counts[labels]
            smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels),
                                        replacement=True, generator=g)
        elif sampler is None or sampler in ("none", "None"):
            shuffle = True
        else:
            raise ValueError(f"sampler không hợp lệ: {sampler}")
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=smp, drop_last=train,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=g,
                      persistent_workers=persistent_workers and num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None)
