# Tổng kết cập nhật hỗ trợ iNaturalist 2018

## ✅ **ĐÃ CẬP NHẬT**

### **1. `src/train/train_gating_map.py`**

**Changes:**
- Thêm `DATASET_CONFIGS_GATING` với config cho cả CIFAR và iNaturalist
- Thêm argument `--dataset` trong CLI
- Thêm argument `--log-file` cho logging
- Auto-load config dựa trên `--dataset`

**Usage:**
```bash
# CIFAR (default)
python -m src.train.train_gating_map --routing dense

# iNaturalist
python -m src.train.train_gating_map --dataset inaturalist2018 --routing dense

# With logging
python -m src.train.train_gating_map --dataset inaturalist2018 --routing dense --log-file logs/gating.log
```

---

### **2. `run_balanced_plugin_gating.py`**

**Changes:**
- Thêm `DATASET_CONFIGS` dictionary
- Thêm `setup_config()` function
- Thêm `--dataset` argument
- Thêm `--log-file` argument cho logging
- Import `argparse`, `sys`, `datetime`

**Usage:**
```bash
# CIFAR (default)
python run_balanced_plugin_gating.py

# iNaturalist
python run_balanced_plugin_gating.py --dataset inaturalist2018

# With logging
python run_balanced_plugin_gating.py --dataset inaturalist2018 --log-file logs/balanced_plugin.log
```

---

### **3. `run_worst_plugin_gating.py`**

**Changes:**
- Tương tự `run_balanced_plugin_gating.py`
- Thêm `DATASET_CONFIGS` dictionary
- Thêm `setup_config()` function
- Thêm `--dataset` argument
- Thêm `--log-file` argument cho logging
- Import `argparse`, `sys`, `datetime`

**Usage:**
```bash
# CIFAR (default)
python run_worst_plugin_gating.py

# iNaturalist
python run_worst_plugin_gating.py --dataset inaturalist2018

# With logging
python run_worst_plugin_gating.py --dataset inaturalist2018 --log-file logs/worst_plugin.log
```

---

## 📊 **Config Chi Tiết**

### **CIFAR-100-LT:**
```python
{
    "splits_dir": "./data/cifar100_lt_if100_splits_fixed",
    "logits_dir": "./outputs/logits/cifar100_lt_if100",
    "gating_checkpoint": "./checkpoints/gating_map/cifar100_lt_if100/final_gating.pth",
    "results_dir": "./results/ltr_plugin/cifar100_lt_if100",
    "expert_names": ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"],
    "num_classes": 100,
    "num_groups": 2,
}
```

### **iNaturalist 2018:**
```python
{
    "splits_dir": "./data/inaturalist2018_splits",
    "logits_dir": "./outputs/logits/inaturalist2018",
    "gating_checkpoint": "./checkpoints/gating_map/inaturalist2018/final_gating.pth",
    "results_dir": "./results/ltr_plugin/inaturalist2018",
    "expert_names": ["ce_baseline"],
    "num_classes": 8142,
    "num_groups": 2,
}
```

**Note:** iNaturalist hiện chỉ có 1 expert (CE) vì chưa train thêm LogitAdjust và BalSoftmax.

---

## 🚀 **Pipeline Đầy Đủ**

### **Bước 1: Generate Splits**
```bash
python scripts/create_inaturalist_splits.py \
    --train-json ./data/train2018.json \
    --val-json ./data/val2018.json \
    --output-dir ./data/inaturalist2018_splits \
    --log-file logs/create_splits.log
```

### **Bước 2: Train CE Expert**
```bash
python train_experts.py \
    --dataset inaturalist2018 \
    --expert ce \
    --log-file logs/expert_ce.log

# Quick test (2 epochs)
python train_experts.py \
    --dataset inaturalist2018 \
    --expert ce \
    --epochs 2 \
    --batch-size 512 \
    --log-file logs/test.log
```

### **Bước 3: Train Gating Network**
```bash
python -m src.train.train_gating_map \
    --dataset inaturalist2018 \
    --routing dense \
    --epochs 100 \
    --log-file logs/gating.log
```

### **Bước 4: Run Plugin Evaluation**

**Balanced Plugin:**
```bash
python run_balanced_plugin_gating.py \
    --dataset inaturalist2018 \
    --log-file logs/balanced_plugin.log
```

**Worst-group Plugin:**
```bash
python run_worst_plugin_gating.py \
    --dataset inaturalist2018 \
    --log-file logs/worst_plugin.log
```

---

## 📝 **Files Changed**

1. ✅ `src/train/train_gating_map.py` (added iNaturalist support)
2. ✅ `run_balanced_plugin_gating.py` (added iNaturalist support)
3. ✅ `run_worst_plugin_gating.py` (added iNaturalist support)

**Previously updated:**
- ✅ `src/train/train_expert.py` (expert training)
- ✅ `train_experts.py` (CLI wrapper)
- ✅ `src/data/inaturalist2018_splits.py` (split generation)
- ✅ `src/models/experts.py` (ResNet-50 backbone)
- ✅ `scripts/create_inaturalist_splits.py` (CLI wrapper)

---

## ⚠️ **Lưu Ý**

### **iNaturalist chỉ có 1 expert**
- Hiện tại chỉ train được CE expert
- Gating network sẽ combine 1 expert (không có ý nghĩa thực tế)
- **TODO**: Train thêm LogitAdjust và BalSoftmax experts

### **Require existing data**
- Phải có `train2018.json` và `val2018.json`
- Phải run `create_inaturalist_splits.py` trước
- Phải train CE expert trước
- Phải export logits từ expert

### **Paths**
- Tất cả paths phải match với config
- Check `splits_dir`, `logits_dir`, `checkpoint_dir` tồn tại
- Create directories nếu cần

---

## ✅ **Verification**

**Test commands:**
```bash
# 1. Check splits exist
ls -lh data/inaturalist2018_splits/*.json

# 2. Check logits exist
ls -lh outputs/logits/inaturalist2018/ce_baseline/*.pt

# 3. Check gating checkpoint (after training)
ls -lh checkpoints/gating_map/inaturalist2018/final_gating.pth

# 4. Dry-run gating training
python -m src.train.train_gating_map --dataset inaturalist2018 --epochs 1 --dry-run
```

---

## 🎉 **Kết Luận**

Bây giờ bạn đã có thể chạy **FULL PIPELINE** cho iNaturalist 2018:

1. ✅ Generate splits
2. ✅ Train experts
3. ✅ Export logits
4. ✅ Train gating network
5. ✅ Run plugin evaluation

**Chỉ còn thiếu:** Train thêm LogitAdjust và BalSoftmax experts để có đủ 3 experts như CIFAR!

