# Files Changed/Added for iNaturalist 2018 Support

## 📁 **NEW FILES CREATED**

### 1. `src/data/inaturalist2018_splits.py` ⭐ NEW
- **Purpose**: Core data loading and split generation for iNaturalist 2018
- **Key Features**:
  - `INaturalistDataset` class: Load images from JSON annotations (COCO format)
  - `get_inaturalist_transforms()`: ImageNet-style transforms (RandomResizedCrop 224, ImageNet normalization)
  - `analyze_inaturalist_distribution()`: Class distribution analysis
  - `visualize_tail_proportion()`: Tail proportion verification (4 plots)
  - `generate_class_distribution_report()`: Detailed reports (MD + CSV + summary)
  - `split_train_for_expert_and_gating()`: 90/10 split preserving imbalance
  - `split_val_into_val_test_tunev()`: 1:1:1 split of validation set
  - `compute_class_weights()`: Compute weights for reweighted metrics
  - Logging support with `TeeOutput`

### 2. `scripts/create_inaturalist_splits.py` ⭐ NEW
- **Purpose**: CLI wrapper for generating iNaturalist splits
- **Usage**: `python scripts/create_inaturalist_splits.py --train-json ... --val-json ...`
- **Features**: argparse CLI, logging support, calls `create_inaturalist2018_splits()`

---

## 📝 **FILES MODIFIED**

### 3. `src/train/train_expert.py` ✏️ MODIFIED
**Major changes for iNaturalist support:**
- Added `EXPERT_CONFIGS_INATURALIST` dictionary (200 epochs, LR 0.4, cosine scheduler)
- Added `DATASET_CONFIGS` dictionary with iNaturalist settings
- Modified `get_dataloaders()`: 
  - Import iNaturalist utilities
  - Create `INaturalistSubset` wrapper for indices
  - Dynamic batch size loading
- Modified `validate_model()`: 
  - Dynamic `num_classes` from config
  - Load `class_to_group` from JSON instead of hardcoded ranges
- Modified `load_class_weights()`: Dynamic `num_classes` instead of hardcoded 100
- Modified `train_single_expert()`:
  - Select configs based on dataset
  - Support cosine annealing with warmup
- Modified `export_logits_for_all_splits()`:
  - Handle both CIFAR and iNaturalist datasets
  - Check for `train_indices.json` existence (skip if not found for iNaturalist)
  - Validate indices for INaturalistSubset
- **Removed reweighted accuracy** from logging and model selection

### 4. `train_experts.py` ✏️ MODIFIED
- Added `--dataset` argument (cifar100_lt_if100 | inaturalist2018)
- Added `--log-file` argument for logging
- Integrated `TeeOutput` class for console + file logging
- Modified `setup_training_environment()`: 
  - Set `CONFIG` based on dataset
  - Apply `--batch-size` override
- Modified `train_single_expert_wrapper()`: Pass overrides to `train_single_expert()`

### 5. `src/models/experts.py` ✏️ MODIFIED
- Modified `Expert.__init__()`:
  - Support `resnet50` backbone for iNaturalist/ImageNet
  - Load `torchvision.models.resnet50` (pretrained=False)
  - Replace classifier with custom Linear layer
  - Dynamic initialization logic

### 6. `src/train/train_gating_map.py` ✏️ MODIFIED
- Added `DATASET_CONFIGS_GATING` dictionary
- Added `--dataset` argument
- Added `--log-file` argument with `TeeOutput`
- Modified `load_labels()`:
  - Load from `*_targets.json` first (iNaturalist)
  - Fallback to CIFAR-style index-based loading
- Modified `load_class_weights()`: Accept `num_classes` as argument
- Updated `main()`: Dynamic config based on dataset

### 7. `run_balanced_plugin_gating.py` ✏️ MODIFIED
- Added `DATASET_CONFIGS` dictionary
- Added `setup_config()` function
- Added `--dataset` argument
- Added `--log-file` argument with `TeeOutput`
- Modified `load_labels()`: Support iNaturalist `*_targets.json`
- Updated `main()`: Call `setup_config()` before proceeding
- Fixed indentation bug in `try-finally` block

### 8. `run_worst_plugin_gating.py` ✏️ MODIFIED
- Added `DATASET_CONFIGS` dictionary
- Added `setup_config()` function
- Added `--dataset` argument
- Added `--log-file` argument with `TeeOutput`
- Modified `load_labels()`: Support iNaturalist `*_targets.json`
- Updated `main()`: Call `setup_config()` before proceeding
- Fixed indentation bug in `try-finally` block

### 9. `src/models/gating.py` ✏️ MODIFIED
- Modified `GatingFeatureBuilder.__call__()`:
  - Handle single-expert case (E=1) for `var()` and `std()` calculations
  - Return zeros for variance/std when only 1 expert exists
  - Use `unbiased=False` to avoid PyTorch warnings

### 10. `README.md` ✏️ MODIFIED
- Added "Dataset Support" section
- Documented iNaturalist 2018 configuration
- Added CLI examples for both datasets
- Updated training commands with dataset arguments
- Added "Quick test" command with 2 epochs

---

## 📊 **SUMMARY**

### Created: **2 new files**
1. `src/data/inaturalist2018_splits.py` (1056 lines)
2. `scripts/create_inaturalist_splits.py` (CLI wrapper)

### Modified: **8 existing files**
1. `src/train/train_expert.py`
2. `train_experts.py`
3. `src/models/experts.py`
4. `src/train/train_gating_map.py`
5. `run_balanced_plugin_gating.py`
6. `run_worst_plugin_gating.py`
7. `src/models/gating.py`
8. `README.md`

### Total: **10 files** (excluding documentation)

---

## 🔑 **KEY TECHNICAL CHANGES**

### Data Loading
- **CIFAR**: `torchvision.datasets.CIFAR100` with subset indices
- **iNaturalist**: Custom `INaturalistDataset` with COCO JSON + `INaturalistSubset` wrapper

### Transforms
- **CIFAR**: `RandomCrop(32, padding=4)` + CIFAR normalization
- **iNaturalist**: `RandomResizedCrop(224)` + ImageNet normalization

### Model Backbone
- **CIFAR**: CIFARResNet-32
- **iNaturalist**: ResNet-50

### Configuration
- All scripts now support `--dataset` argument
- Dynamic `num_classes` (100 vs 8142)
- Dataset-specific expert configs (schedulers, epochs, batch size)

### Logging
- `TeeOutput` class added to all main scripts
- Logs saved to specified file while displaying to console

