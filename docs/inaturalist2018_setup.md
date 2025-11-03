# iNaturalist 2018 Dataset Setup Guide

## Overview

The iNaturalist 2018 dataset is a large-scale fine-grained classification dataset with:
- **437,513** training images across **8,142** species classes
- **24,426** validation images
- Inherent long-tail distribution (no need to create it)
- Natural threshold at ~20 samples distinguishes head/tail classes

## Dataset Download

Download the iNaturalist 2018 dataset from:
- Official: https://github.com/visipedia/inat_comp

You'll need:
1. Training images: `train_val2018/` directory
2. `train.json`: COCO-format annotations for training set
3. `val.json`: COCO-format annotations for validation set

Expected structure:
```
data/
  inaturalist2018/
    train_val2018/          # Image directory
      *.jpg                 # Images
    train.json              # Training annotations (COCO format)
    val.json                # Validation annotations (COCO format)
```

## COCO Format

The JSON files should follow COCO annotation format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "00001.jpg",
      ...
    }
  ],
  "annotations": [
    {
      "image_id": 1,
      "category_id": 0,
      ...
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "species_1",
      ...
    }
  ]
}
```

## Generating Splits

### Step 1: Run Split Generator

**Basic usage (without logging):**
```bash
python scripts/create_inaturalist_splits.py \
  --train-json data/train2018.json \
  --val-json data/val2018.json \
  --data-dir data \
  --output-dir data/inaturalist2018_splits
```

**With logging (recommended for server runs):**
```bash
python scripts/create_inaturalist_splits.py \
  --train-json data/train2018.json \
  --val-json data/val2018.json \
  --data-dir data/ \
  --output-dir data/inaturalist2018_splits \
  --log-file logs/inaturalist_splits_$(date +%Y%m%d_%H%M%S).log
```

**Or in a single line (to avoid bash line break issues):**
```bash
python scripts/create_inaturalist_splits.py --train-json data/train2018.json --val-json data/val2018.json --data-dir data/inaturalist2018 --output-dir data/inaturalist2018_splits --log-file logs/inaturalist_splits.log
```

### Step 2: Check Output

The script will generate:

1. **JSON Files** in `data/inaturalist2018_splits/`:
   - `expert_indices.json` - Indices for expert training (90%)
   - `gating_indices.json` - Indices for gating training (10%)
   - `val_indices.json` - Validation indices (1/3 of original val)
   - `test_indices.json` - Test indices (1/3 of original val)
   - `tunev_indices.json` - Tuning indices (1/3 of original val)
   - `train_class_counts.json` - Samples per class in training
   - `class_weights.json` - Importance weights for reweighting

2. **Visualizations**:
   - `distribution_last_100_classes.png` - Distribution of 100 least-populated classes
   - `expert_gating_split_last_100_classes.png` - Expert/Gating split visualization

## Split Details

### Training Splits (Maintains Long-Tail)

- **Expert**: 90% of training data (~393,762 samples)
- **Gating**: 10% of training data (~43,751 samples)
- Both splits maintain the **same imbalance ratio**
- Distribution preserves head/tail structure

### Validation Splits (1:1:1)

Due to limited validation samples (24k), we use 1:1:1 split:
- **Val**: ~8,142 samples
- **Test**: ~8,142 samples
- **TuneV**: ~8,142 samples

All validation splits are **balanced** (disjoint, no duplication).

## Head/Tail Classification

Classes are divided based on training samples:
- **Head**: Classes with > 20 training samples
- **Tail**: Classes with ≤ 20 training samples

This threshold ensures:
- Head classes have sufficient data for reliable training
- Tail classes represent the truly underrepresented species
- Balanced evaluation across both groups

## Class Weights

Importance weights are computed for metric reweighting:
```python
weights[i] = P_train(class i) * num_classes
```

This allows test set to reflect training distribution for fair evaluation.

## Usage in Training

Once splits are generated, you can use them in your training scripts:

```python
import json
from pathlib import Path

# Load splits
splits_dir = Path("data/inaturalist2018_splits")
expert_indices = json.load(open(splits_dir / "expert_indices.json"))
class_weights = json.load(open(splits_dir / "class_weights.json"))

# Use in dataloader
# ... your training code
```

## Troubleshooting

### Issue: JSON format mismatch
**Solution**: Verify your JSON files follow COCO format with `images`, `annotations`, and `categories` keys.

### Issue: Images not found
**Solution**: Check that `--data-dir` points to the correct image directory and file_name paths in JSON match actual image locations.

### Issue: Too many classes to visualize
**Solution**: The script only visualizes the last 100 classes by default. Adjust with `--visualize-n`.

## Training Configuration

For iNaturalist 2018, use the following configuration (from paper Table 3):

### Expert Models
- **Model**: ResNet-50 (from torchvision)
- **Batch Size**: 1024
- **Learning Rate**: 0.4
- **Scheduler**: Cosine annealing
- **Warmup**: 5 epochs
- **Decay**: 0.1 at epochs 45, 100, 150
- **Epochs**: 200
- **Optimizer**: SGD with momentum=0.9, weight_decay=1e-4

### Example Training Configuration
```python
from src.models.experts import Expert

# Create ResNet-50 expert for iNaturalist
expert = Expert(
    num_classes=8142,
    backbone_name='resnet50',
    dropout_rate=0.0,
    init_weights=True
)
```

## Next Steps

After generating splits:
1. Train experts on `expert` split with ResNet-50 (batch=1024, cosine scheduler)
2. Train gating network on `gating` split  
3. Run LtR plugin evaluation
4. Compute metrics with importance weighting

## References

- iNaturalist Competition 2018: https://github.com/visipedia/inat_comp
- Original paper on iNaturalist: Van Horn et al., CVPR 2018

