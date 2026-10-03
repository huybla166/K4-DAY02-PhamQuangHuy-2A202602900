#!/bin/bash
# run_all.sh - chạy lại TOÀN BỘ bài lab theo đúng thứ tự đã làm (từ thư mục gốc repo: có eval.py, data/).
# Mọi thí nghiệm đi qua một hàm train.run(Config); bảng thí nghiệm ở code/experiments.py.
# Bỏ qua lần chạy đã có runs/<exp>/seed<k>/summary.json (chạy tiếp được khi bị ngắt).
set -e
export PYTHONUTF8=1 PYTHONWARNINGS=ignore
PY=${PY:-python}
S=submissions/2A202602900_pham_quang_huy; C=$S/code; P=$S/predictions; L=data/labels; E=$S/logs/eval
W=${WORKERS:-4}

# Bước 0: test code, EDA + kiểm tra chia dữ liệu, sanity check pipeline, ước lượng ngân sách GPU
$PY -m unittest $C/tests_code.py
$PY $C/eda.py --out $S
$PY $C/temporal_overlap.py                         # mức 'cùng buổi chụp' giữa train và val/test (chỉ dùng tên file)
$PY $C/sanity_checks.py --out $S --backbones resnet50 convnext_tiny
$PY $C/probe_speed.py --out $S/logs/probe_speed.json --steps 15

# Bước 1: 7 backbone, công thức nền T00, seed 0 (+ độ trễ sơ bộ batch 1)
$PY $C/experiments.py --ids B01 B02 B03 B04 B05 B06 B07 --latency --workers $W

# Bước 2: trên ConvNeXt-T (chọn ở Bước 1). Nền T00 với 3 seed để đo nhiễu, rồi mỗi thí nghiệm khác T00 một yếu tố
$PY $C/experiments.py --ids T00 --seeds 0 1 2 --workers $W
$PY $C/experiments.py --ids T06 T08 T09 T10 T13 T02 T01 T03 T04 T05 T11 T07 T12 T14 T15 --workers $W
# kết hợp các yếu tố tốt (chọn trên val), rồi chốt công thức chung kết (macro-F1 val cao nhất trong T13/T07/C01/C02)
$PY $C/experiments.py --ids C01 C02 --workers $W
$PY $C/choose_final.py

# Bước 4a: huấn luyện chung kết F01 với 3 seed (chưa đụng test)
$PY $C/experiments.py --ids F01 --seeds 0 1 2 --workers $W

# Bước 3: phương pháp suy luận trên VAL của F01 seed 0 (+ ensemble khác kiến trúc, EMA, gộp BN, FP16) + độ trễ
$PY $C/inference_exps.py --main runs/F01/seed0 --ensemble runs/B05/seed0 runs/B04/seed0 --ema-run runs/F01/seed0 \
    --bn-run runs/B01/seed0
METHOD=$($PY $C/choose_method.py | tail -1)      # quy tắc chọn trên val, ghi code/final_method.json

# Bước 4b: TEST đúng MỘT lần mỗi seed (final.py từ chối ghi đè), chung kết + mốc T00/I00
$PY $C/final.py --exp F01 --seeds 0 1 2 --method $METHOD --ts
$PY $C/final.py --exp T00 --seeds 0 1 2 --method I00
mkdir -p $E
for g in F01 F01_uncal T00; do
  $PY eval.py score --pred "$P/${g}_seed*_test.csv" --test-csv $L/test_subset0.csv --labels $L/labels.csv --tag $g --out $E
done
P95=$($PY -c "import json;j=json.load(open('$C/final_method.json',encoding='utf-8'));print(round(min(j['p95_ms_amp'],j['p95_ms_fp32']),2))")
$PY eval.py grade --final "$P/F01_seed*_test.csv" --baseline "$P/T00_seed*_test.csv" --uncal "$P/F01_uncal_seed*_test.csv" \
    --final-val "$P/F01_seed*_val.csv" --latency-p95-ms $P95 --latency-method proper --test-csv $L/test_subset0.csv \
    --val-csv $L/val_subset0.csv --labels $L/labels.csv --out $E

# Điểm thưởng: cùng cấu hình trên fold 1, 2 (bộ ba file của fold đó), seed 0
for k in 1 2; do
  $PY $C/experiments.py --ids F01_f$k --workers $W
  $PY $C/final.py --exp F01_f$k --seeds 0 --method $METHOD --ts --fold $k
  $PY eval.py score --pred "$P/F01_f${k}_seed*_test.csv" --test-csv $L/test_subset$k.csv --labels $L/labels.csv --tag F01_f$k --out $E
done
# Bước 5: phân tích lỗi (ma trận nhầm lẫn, ảnh sai, Grad-CAM), lệch phân phối + thích ứng BN (trên val)
$PY $C/error_analysis.py --final "$P/F01_seed*_test.csv" --baseline "$P/T00_seed*_test.csv" --final-run runs/F01/seed0 \
    --method $METHOD --bn-run runs/B01/seed0
$PY $C/photometric_errors.py --pred "$P/F01_seed*_test.csv"   # lỗi theo độ sáng/màu/tương phản
# chẩn đoán Bước 1 (thêm sau khi thấy CNN có BN underfit) + linear probe DINOv2 (thưởng)
$PY $C/experiments.py --ids B08 B09 --latency --workers $W
$PY $C/dinov2_probe.py
# bảng results.xlsx + biểu đồ tổng hợp
$PY $C/make_results.py --final F01
