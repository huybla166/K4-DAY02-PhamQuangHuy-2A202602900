"""build_notebook.py - sinh code/lab_day2.ipynb: notebook chạy lại toàn bộ bài lab trên Colab/Kaggle/máy cục bộ.

Notebook gọi đúng các script trong code/ theo thứ tự của code/run_all.sh (một hàm train.run dùng chung).
python submissions/<bai>/code/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/huybla166/K4-Track4-Day2-Deeplearning-Advance.git"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


def show(*names):
    files = ", ".join(f'"{n}"' for n in names)
    return code(f"""
from IPython.display import Image, display
for f in [{files}]:
    p = f"{{S}}/figures/{{f}}.png"
    if os.path.exists(p): display(Image(p))
""")


CELLS = [
    md("""
# Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds (bài làm của Phạm Quang Huy)

Notebook chạy lại **toàn bộ** bài lab theo đúng thứ tự đã làm (giống `code/run_all.sh`). Mọi thí nghiệm đi qua
một hàm `train.run(Config)` (`code/train.py`); bảng thí nghiệm và khác biệt so với nền ở `code/experiments.py`.

* Chạy được trên **Kaggle** (GPU T4/P100, bật Internet), **Colab** (GPU T4) hoặc máy có GPU NVIDIA.
* Số liệu trong `report.md` được chạy trên RTX 3060 Laptop 6 GB; trên GPU khác thời gian/độ trễ sẽ khác,
  độ chính xác cùng mức (cuDNN không tất định nên không trùng từng bit).
* **Quy tắc:** chọn mọi thứ trên **val**; test chỉ chạy **một lần mỗi seed** ở Bước 4 (`code/final.py` từ chối ghi đè).
* Bỏ qua lần chạy đã có `runs/<exp>/seed<k>/summary.json`, nên chạy lại ô sau khi phiên bị ngắt sẽ tiếp tục.
"""),
    md("## 0. Cài đặt, tải dữ liệu"),
    code(f"""
import os, sys, subprocess, platform
REPO_URL = "{REPO_URL}"
REPO_DIR = "K4-Track4-Day2-Deeplearning-Advance"
if not os.path.exists("eval.py"):
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
    os.chdir(REPO_DIR)
S = "submissions/2A202602900_pham_quang_huy"     # thư mục bài nộp
C = f"{{S}}/code"; P = f"{{S}}/predictions"; L = "data/labels"; E = f"{{S}}/logs/eval"
os.environ["PYTHONUTF8"] = "1"; os.environ["PYTHONWARNINGS"] = "ignore"
print(os.getcwd(), platform.python_version())
"""),
    code("""
!pip -q install timm openpyxl fvcore scikit-learn
import torch, timm
print("torch", torch.__version__, "| timm", timm.__version__, "| GPU:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU")
"""),
    code("""
import hashlib, zipfile, urllib.request
os.makedirs("data/labels", exist_ok=True); os.makedirs("data/images", exist_ok=True)
if not os.path.exists("data/images.zip"):
    urllib.request.urlretrieve("https://zenodo.org/records/7939060/files/images.zip?download=1", "data/images.zip")
h = hashlib.md5()
with open("data/images.zip", "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
assert h.hexdigest() == "b7b30f96d466fba86016aa5a26606e0f", f"MD5 sai: {h.hexdigest()}"
if len(os.listdir("data/images")) < 17509:
    zipfile.ZipFile("data/images.zip").extractall("data/images")
BASE = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"
for k in range(5):            # fold 0 dùng chính; fold 1, 2 chỉ cho điểm thưởng nhiều fold
    for sp in ("train", "val", "test"):
        urllib.request.urlretrieve(f"{BASE}/{sp}_subset{k}.csv", f"data/labels/{sp}_subset{k}.csv")
urllib.request.urlretrieve(f"{BASE}/labels.csv", "data/labels/labels.csv")
print(len(os.listdir("data/images")), "ảnh;", sorted(os.listdir("data/labels")))
"""),
    md("## Bước 0 — Test code, EDA + kiểm tra chia dữ liệu (S1–S4), sanity check pipeline"),
    code("""
!python -m unittest {C}/tests_code.py
!python {C}/eda.py --out {S}
!python {C}/temporal_overlap.py
!python {C}/sanity_checks.py --out {S} --backbones resnet50 convnext_tiny
!python {C}/probe_speed.py --out {S}/logs/probe_speed.json --steps 15
"""),
    show("eda_class_distribution", "eda_samples", "sanity_overfit", "sanity_augmentations", "sanity_mix"),
    md("## Bước 1 — 7 backbone với công thức nền T00 (seed 0, 12 epoch) + độ trễ sơ bộ"),
    code("!python {C}/experiments.py --ids B01 B02 B03 B04 B05 B06 B07 --latency --workers 4"),
    md("## Bước 2 — Công thức huấn luyện trên ConvNeXt-T: T00 × 3 seed (đo nhiễu), mỗi thí nghiệm khác T00 một yếu tố, rồi kết hợp"),
    code("""
!python {C}/experiments.py --ids T00 --seeds 0 1 2 --workers 4
!python {C}/experiments.py --ids T06 T08 T09 T10 T13 T02 T01 T03 T04 T05 T11 T07 T12 T14 T15 --workers 4
!python {C}/experiments.py --ids C01 C02 --workers 4
!python {C}/choose_final.py
"""),
    md("## Bước 4a — Huấn luyện chung kết F01 với 3 seed (chưa đụng test)"),
    code("!python {C}/experiments.py --ids F01 --seeds 0 1 2 --workers 4"),
    md("## Bước 3 — Phương pháp suy luận trên **val** của F01 seed 0 + độ trễ p50/p95/p99; chọn phương pháp theo quy tắc"),
    code("""
!python {C}/inference_exps.py --main runs/F01/seed0 --ensemble runs/B05/seed0 runs/B04/seed0 --ema-run runs/F01/seed0 --bn-run runs/B01/seed0
METHOD = subprocess.run([sys.executable, f"{C}/choose_method.py"], capture_output=True, text=True).stdout.split()[-1]
print("phương pháp suy luận chung kết:", METHOD)
"""),
    show("inference_tradeoff", "reliability_val"),
    md("## Bước 4b — TEST đúng một lần mỗi seed (chung kết + mốc T00/I00), chấm bằng `eval.py`"),
    code("""
!python {C}/final.py --exp F01 --seeds 0 1 2 --method {METHOD} --ts
!python {C}/final.py --exp T00 --seeds 0 1 2 --method I00
for g in ["F01", "F01_uncal", "T00"]:
    !python eval.py score --pred "{P}/{g}_seed*_test.csv" --test-csv {L}/test_subset0.csv --labels {L}/labels.csv --tag {g} --out {E}
import json
j = json.load(open(f"{C}/final_method.json", encoding="utf-8")); P95 = round(min(j["p95_ms_amp"], j["p95_ms_fp32"]), 2)
!python eval.py grade --final "{P}/F01_seed*_test.csv" --baseline "{P}/T00_seed*_test.csv" --uncal "{P}/F01_uncal_seed*_test.csv" --final-val "{P}/F01_seed*_val.csv" --latency-p95-ms {P95} --latency-method proper --test-csv {L}/test_subset0.csv --val-csv {L}/val_subset0.csv --labels {L}/labels.csv --out {E}
"""),
    md("## Điểm thưởng — cùng cấu hình trên fold 1, 2; chẩn đoán B08/B09; DINOv2 linear probe"),
    code("""
for k in (1, 2):
    !python {C}/experiments.py --ids F01_f{k} --workers 4
    !python {C}/final.py --exp F01_f{k} --seeds 0 --method {METHOD} --ts --fold {k}
    !python eval.py score --pred "{P}/F01_f{k}_seed*_test.csv" --test-csv {L}/test_subset{k}.csv --labels {L}/labels.csv --tag F01_f{k} --out {E}
!python {C}/experiments.py --ids B08 B09 --latency --workers 4
!python {C}/dinov2_probe.py
"""),
    md("## Bước 5 — Phân tích lỗi, lệch phân phối, `results.xlsx`"),
    code("""
!python {C}/error_analysis.py --final "{P}/F01_seed*_test.csv" --baseline "{P}/T00_seed*_test.csv" --final-run runs/F01/seed0 --method {METHOD} --bn-run runs/B01/seed0
!python {C}/photometric_errors.py --pred "{P}/F01_seed*_test.csv"
!python {C}/make_results.py --final F01
"""),
    show("confusion_test", "misclassified_test", "gradcam_misclassified", "backbones_tradeoff", "training_ablation"),
]


def main():
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    (HERE / "lab_day2.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("đã ghi", HERE / "lab_day2.ipynb")


if __name__ == "__main__":
    main()
