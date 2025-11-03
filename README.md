# capstone_l2r_ess

**Learning To Reject meets MoE in Long Tail Learning**

## 🎯 Vision

Combining 3 experts (CE, LogitAdjust, BalancedSoftmax) + gating router + plug-in rejector rule to solve selective classification challenges in long-tail learning: (i) bias & miscalibration in tail, (ii) head/tail tradeoff with global threshold, (iii) instability with few samples.

## Dataset Support

This project supports:
- **CIFAR-100-LT**: Long-tail version of CIFAR-100 with imbalance factor 100
  - Model: CIFARResNet-32 (CIFAR-optimized)
  - Batch size: 128
  
- **iNaturalist 2018**: Large-scale fine-grained classification with 8,000+ classes and inherent long-tail distribution
  - Model: ResNet-50 (ImageNet architecture)
  - Batch size: 1024
  - Scheduler: Cosine annealing with warmup

### Dataset Setup

#### CIFAR-100-LT
```bash
# Generate CIFAR-100-LT splits
python src/data/balanced_test_splits.py
```

#### iNaturalist 2018
```bash
# Generate iNaturalist 2018 splits from JSON files
python scripts/create_inaturalist_splits.py \
  --train-json /path/to/train.json \
  --val-json /path/to/val.json \
  --data-dir data/inaturalist2018/train_val2018 \
  --log-file logs/inaturalist2018_splits_$(date +%Y%m%d_%H%M%S).log
```

## Run pipeline

#### Step 1: Train 3 Experts

Train 3 experts with different long-tail strategies:

**For CIFAR-100-LT:**
```bash
python train_experts.py --dataset cifar100_lt_if100 --log-file logs/experts_cifar.log
```

**For iNaturalist 2018:**
```bash
python train_experts.py --dataset inaturalist2018 --log-file logs/experts_inat.log
```

**Quick test (2 epochs, batch 512):**
```bash
python train_experts.py --dataset inaturalist2018 --expert ce --epochs 2 --batch-size 512 --log-file logs/inat_test.log
```

#### Step 2: Train Gating Network

Train Mixture of Experts (MoE) router:

```bash
python -m src.train.train_gating_map --routing dense
```

#### Step 3: Train LtR Plugin (Main Method)

**Balanced Objective** (Algorithm 1 - Power Iteration):

```bash
python run_balanced_plugin_gating.py
```

**Worst-group Objective** (Algorithm 2 - Exponentiated Gradient):

```bash
python run_worst_plugin_gating.py
```

### Reproduce paper results

**CE-only Plugin** (Single expert, no gating):

```bash
python run_balanced_plugin_ce_only.py
```

```bash
python run_worst_plugin_ce_only.py
```

#### Note

Run with only CE

```
python -m src.train.train_ce_expert_paper_final
```
