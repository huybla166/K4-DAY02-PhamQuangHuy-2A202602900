# Lab Day 2 — DeepWeeds: backbone, công thức huấn luyện, suy luận (bài làm của Phạm Quang Huy - MSSV: 2A202602900)

**Kết quả chính** (test fold 0, 3 seed, tính lại bằng `eval.py`): cấu hình chung kết F01 = ConvNeXt-T `in12k_ft_in1k` + EMA + Mixup + lật dọc/xoay 90°, suy luận 1 view ảnh đầy đủ 256 px + temperature scaling → **macro-F1 0,9770 ± 0,0008, top-1 98,19 ± 0,12%**, recall Chinee/Snake 94,2% / 96,1%, ECE 0,0041; mốc T00 + I00: 0,9749 ± 0,0010. Độ trễ p95 batch 1 = 7,2 ms (FP32, RTX 3060 Laptop). `eval.py grade`: 19/20 điểm phần I (đề xuất).

| Sản phẩm | Ở đâu |
|---|---|
| Báo cáo kết luận | [`report.md`](report.md) |
| Bảng so sánh mọi thí nghiệm | [`results.xlsx`](results.xlsx) (sheet `Summary`, `Backbones`, `Training`, `Inference`, `Final`, `PerClass`, `Latency`, `Bonus`) |
| Biểu đồ training, mỗi `exp_id` một ảnh | [`curves/`](curves) |
| File dự đoán test (chung kết + mốc, mọi seed), val và bản chưa temperature scaling | [`predictions/`](predictions) |
| Toàn bộ code (bộ khung `starter/` đã hoàn thiện + script thí nghiệm) | [`code/`](code) |
| Notebook chạy lại toàn bộ | [`code/lab_day2.ipynb`](code/lab_day2.ipynb) — mở trên Colab: <https://colab.research.google.com/github/huybla166/K4-Track4-Day2-Deeplearning-Advance/blob/main/submissions/2A202602900_pham_quang_huy/code/lab_day2.ipynb> (link hoạt động sau khi thư mục này được push lên nhánh `main` của fork) |
| Log từng lần chạy (config, history theo epoch, summary, LR theo bước) | [`logs/<exp_id>/seed<k>/`](logs) ; suy luận: `logs/inference/`, chung kết: `logs/final/`, phân tích: `logs/analysis/`, thưởng: `logs/bonus/` |
| Ảnh EDA, sanity check, phân tích | [`figures/`](figures) |

Checkpoint (~100 MB mỗi lần chạy) **không** commit (xem `.gitignore` của repo); chạy lại notebook sẽ sinh lại.

## Môi trường đã dùng để ra số liệu trong báo cáo

| Thành phần | Phiên bản |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB, driver 595.79 (Windows 11, WDDM) |
| Python | 3.11.9 |
| PyTorch / CUDA / cuDNN | 2.14.1+cu130 / 13.0 / 9.24 |
| torchvision | 0.29.1+cu130 |
| timm | 1.0.30 |
| numpy / pandas / scikit-learn / scipy | 2.4.6 / 3.0.6 / 1.9.1 / 1.17.1 |
| matplotlib / openpyxl / Pillow / fvcore | 3.11.2 / 3.1.5 / 12.3.0 / 0.1.5.post20221221 |

Mỗi lần chạy ghi lại phiên bản thư viện, GPU, tag trọng số `timm`, mean/std chuẩn hoá và kết quả kiểm tra chia dữ liệu vào `logs/<exp_id>/seed<k>/config.json`.

**Seed:** mọi thí nghiệm sàng (Bước 1, 2) dùng seed 0. Mốc `T00` và chung kết `F01` dùng seed 0, 1, 2. Seed cố định `random`, `numpy`, `torch` (CPU + CUDA) và generator của DataLoader (thứ tự batch, augmentation); seed **không** đổi cách chia (fold 0 cố định). `cudnn.benchmark = True` nên hai lần chạy cùng seed không trùng bit (đo được: B03 và T00 seed 0 cùng cấu hình, macro-F1 val 0,9705 và 0,9714).

## Cách chạy lại

Chạy từ **thư mục gốc repo** (nơi có `eval.py`, `data/`). Dữ liệu: `data/images/*.jpg` (giải nén `images.zip` từ Zenodo, MD5 `b7b30f96d466fba86016aa5a26606e0f`) và `data/labels/*.csv` (GitHub của tác giả). Notebook làm sẵn bước tải.

```bash
pip install torch torchvision timm scikit-learn scipy openpyxl matplotlib fvcore   # cùng phiên bản như bảng trên nếu muốn khớp số
export PYTHONUTF8=1                       # Windows: eval.py/log tiếng Việt cần UTF-8
# tuỳ chọn khi mạng chậm: tải trước trọng số rồi trỏ TIMM_WEIGHTS_DIR (cùng file trên HuggingFace)
# bash weights/fetch_weights.sh resnet50.tv_in1k vit_small_patch14_dinov2.lvd142m ... ; export TIMM_WEIGHTS_DIR=weights
bash submissions/2A202602900_pham_quang_huy/code/run_all.sh    # toàn bộ, đúng thứ tự dưới đây
```

`code/run_all.sh` chạy đúng thứ tự đã làm (mọi thí nghiệm đi qua **một** hàm `train.run(Config)`; danh sách thí nghiệm và khác biệt so với nền nằm ở `code/experiments.py`):

1. **Bước 0** — 20 unit test (`tests_code.py`); EDA và kiểm tra chia dữ liệu (`eda.py`); mức trùng buổi chụp train/test (`temporal_overlap.py`); sanity check: loss ban đầu, overfit 16 ảnh, ảnh sau augmentation (`sanity_checks.py`); ước lượng tốc độ (`probe_speed.py`).
2. **Bước 1** — `experiments.py --ids B01 … B07 --latency` (seed 0).
3. **Bước 2** — `experiments.py --ids T00 --seeds 0 1 2`, rồi T01–T15 và C01, C02 (seed 0); `choose_final.py` chốt công thức chung kết (= C02).
4. **Bước 4a** — `experiments.py --ids F01 --seeds 0 1 2` (chưa đụng test).
5. **Bước 3** — `inference_exps.py --main runs/F01/seed0 …` trên **val**; `choose_method.py` chốt phương pháp (= I04_256 + TS).
6. **Bước 4b** — `final.py --exp F01 --seeds 0 1 2 --method I04_256 --ts` và `final.py --exp T00 --seeds 0 1 2 --method I00`. Mỗi seed test đúng một lần. Sau đó chạy `eval.py score` và `eval.py grade` (kết quả ở `logs/eval/`).
7. **Thưởng và phân tích** — F01 trên fold 1, 2; `error_analysis.py` (ma trận nhầm lẫn, ảnh sai, Grad-CAM, lệch phân phối, thích ứng BN); `photometric_errors.py`; B08/B09 (chẩn đoán); `dinov2_probe.py`.
8. **Bảng** — `make_results.py --final F01` sinh `results.xlsx` và các biểu đồ tổng hợp.

Thời gian đo được trên RTX 3060 Laptop: ConvNeXt-T 12 epoch ≈ 9–10 phút/lần chạy; 34 lần huấn luyện ≈ 5,5 giờ GPU, cộng ~1 giờ cho suy luận, phân tích và đo độ trễ.

## Cấu trúc `code/`

| File | Vai trò |
|---|---|
| `dataset.py` | đọc CSV fold, `check_split` (S1–S4: số ảnh, giao rỗng, hợp 17.509, file tồn tại), transform/augmentation, `DeepWeedsDataset`, `make_loader`, `PreloadedEvalLoader` |
| `model.py` | backbone `timm`, khởi tạo (từ đầu / đóng băng / tinh chỉnh), 3 nhóm tham số (không weight decay cho norm/bias), đếm params/GMAC |
| `losses.py` | CE, label smoothing, focal, CE có trọng số lớp, Mixup/CutMix (trộn cả nhãn) |
| `train.py` | `Config` + `run(cfg)` dùng chung: AMP, warmup + cosine theo bước, EMA, chọn checkpoint theo macro-F1 val, lưu logit, vẽ đường cong |
| `inference.py` | TTA (lật, D4, multi-crop, đa tỉ lệ), gộp xác suất/logit, ensemble, temperature scaling, gộp BatchNorm |
| `benchmark.py` | đo độ trễ đúng cách: warmup 10, `cuda.synchronize`, ≥ 50 lần, p50/p95/p99 |
| `experiments.py` | bảng thí nghiệm B/T/C/F (khác nền ở đâu) + trình chạy hàng loạt |
| `methods.py`, `inference_exps.py` | Bước 3: các phương pháp suy luận trên val + độ trễ |
| `final.py` | Bước 4: test **đúng một lần mỗi seed**, ghi `predictions/` bằng `eval.save_predictions`, từ chối ghi đè |
| `make_results.py` | sinh `results.xlsx` và biểu đồ tổng hợp từ log thật |
| `error_analysis.py`, `photometric_errors.py` | ma trận nhầm lẫn, ảnh đoán sai, Grad-CAM, lệch phân phối + thích ứng BN (trên val); tỉ lệ lỗi theo độ sáng/màu/tương phản |
| `eda.py`, `sanity_checks.py`, `probe_speed.py`, `temporal_overlap.py`, `dinov2_probe.py` | EDA, kiểm tra pipeline, ước lượng ngân sách GPU, mức trùng buổi chụp train/test, linear probe DINOv2 (thưởng) |
| `choose_final.py`, `choose_method.py` | quy tắc chọn công thức chung kết và phương pháp suy luận (chỉ trên val), ghi `final_choice.json`, `final_method.json` |
| `tests_code.py` | 20 kiểm tra tự viết (focal γ=0 ≡ CE, CutMix, gộp BN, EMA, lịch LR, đóng băng giữ BN eval, ba đường suy luận cho cùng tensor, ...) |
| `build_notebook.py`, `run_all.sh` | sinh notebook chạy lại; lệnh chạy toàn bộ |

`eval.py` của repo được dùng nguyên bản (không sửa).
