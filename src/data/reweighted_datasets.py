#!/usr/bin/env python3
"""
Reweighted data preparation for CIFAR-100-LT WITHOUT duplication.
Implements the methodology:
1. Train: Standard exponential profile long-tail (UNCHANGED)
2. Val/Test/TuneV: Split from balanced CIFAR-100 test set (NO duplication)
3. Metrics: Reweight by train distribution when computing (NOT in data)

This approach:
- Avoids data leakage from duplication
- Val, TuneV, Test are all disjoint
- Reweighting happens at metric computation time
"""

import numpy as np
import json
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset

class CIFAR100LTDataset(Dataset):
    """Custom Dataset wrapper for CIFAR-100-LT with flexible indexing."""
    
    def __init__(self, cifar_dataset, indices, transform=None):
        self.cifar_dataset = cifar_dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map to actual CIFAR index
        cifar_idx = self.indices[idx]
        image, label = self.cifar_dataset[cifar_idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_cifar100_transforms():
    """Get CIFAR-100 transforms following paper specifications."""
    
    # Training transforms (basic augmentation as per Menon et al., 2021a)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 mean
            std=[0.2675, 0.2565, 0.2761]   # CIFAR-100 std
        )
    ])
    
    # Evaluation transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    return train_transform, eval_transform

def get_exponential_counts(num_classes: int = 100, imb_factor: float = 100, max_samples: int = 500) -> List[int]:
    """Generate exponential profile counts for long-tail distribution."""
    counts = []
    for cls_idx in range(num_classes):
        # Exponential decay: n_i = n_max * (IF)^(-i/(C-1))
        num = max_samples * (imb_factor ** (-cls_idx / (num_classes - 1.0)))
        counts.append(max(1, int(num)))
    return counts

def create_longtail_train(cifar_train_dataset, imb_factor: float = 100, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """Create long-tail training set using exponential profile."""
    print(f"Creating CIFAR-100-LT training set (IF={imb_factor})...")
    
    np.random.seed(seed)
    targets = np.array(cifar_train_dataset.targets)
    num_classes = 100
    
    # Get target counts
    target_counts = get_exponential_counts(num_classes, imb_factor, 500)
    
    # Sample indices for each class
    train_indices = []
    actual_counts = []
    
    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        num_to_sample = min(target_counts[cls], len(cls_indices))
        
        # Random sample without replacement
        sampled = np.random.choice(cls_indices, num_to_sample, replace=False)
        train_indices.extend(sampled.tolist())
        actual_counts.append(num_to_sample)
    
    train_targets = targets[train_indices].tolist()
    
    print(f"  Created training set: {len(train_indices):,} samples")
    print(f"  Head class: {actual_counts[0]} samples")
    print(f"  Tail class: {actual_counts[-1]} samples")
    print(f"  Actual IF: {actual_counts[0] / actual_counts[-1]:.1f}")
    
    return train_indices, train_targets, actual_counts

def create_balanced_test_splits(
    cifar_test_dataset,
    val_ratio: float = 0.15,
    tunev_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
    """
    Split the balanced CIFAR-100 test set into Val, TuneV, and Test.
    
    Original CIFAR-100 test: 10,000 samples (100 per class)
    Split strategy:
    - Val: 15% = 1,500 samples (15 per class)
    - TuneV: 10% = 1,000 samples (10 per class)  
    - Test: 75% = 7,500 samples (75 per class)
    
    All splits are BALANCED and DISJOINT (no data leakage).
    
    Args:
        cifar_test_dataset: Original CIFAR-100 test dataset
        val_ratio: Proportion for validation (default 0.15)
        tunev_ratio: Proportion for tuneV (default 0.10)
        seed: Random seed
        
    Returns:
        Tuple of (val_indices, val_targets, tunev_indices, tunev_targets, 
                  test_indices, test_targets)
    """
    print("\nCreating balanced Val/TuneV/Test splits from CIFAR-100 test...")
    
    np.random.seed(seed)
    test_targets = np.array(cifar_test_dataset.targets)
    num_classes = 100
    samples_per_class = 100  # CIFAR-100 test has 100 samples per class
    
    # Calculate samples per class for each split
    val_per_class = int(samples_per_class * val_ratio)
    tunev_per_class = int(samples_per_class * tunev_ratio)
    test_per_class = samples_per_class - val_per_class - tunev_per_class
    
    print(f"  Val: {val_per_class} samples per class")
    print(f"  TuneV: {tunev_per_class} samples per class")
    print(f"  Test: {test_per_class} samples per class")
    
    val_indices = []
    tunev_indices = []
    test_indices = []
    
    # Split each class independently to ensure balance
    for cls in range(num_classes):
        cls_indices = np.where(test_targets == cls)[0]
        np.random.shuffle(cls_indices)
        
        # Split: val | tunev | test
        val_end = val_per_class
        tunev_end = val_end + tunev_per_class
        
        val_indices.extend(cls_indices[:val_end].tolist())
        tunev_indices.extend(cls_indices[val_end:tunev_end].tolist())
        test_indices.extend(cls_indices[tunev_end:].tolist())
    
    # Get corresponding targets
    val_targets = test_targets[val_indices].tolist()
    tunev_targets = test_targets[tunev_indices].tolist()
    test_targets_final = test_targets[test_indices].tolist()
    
    print("\n  Final splits (all BALANCED):")
    print(f"    Validation: {len(val_indices):,} samples")
    print(f"    TuneV: {len(tunev_indices):,} samples")
    print(f"    Test: {len(test_indices):,} samples")
    print(f"    Total: {len(val_indices) + len(tunev_indices) + len(test_indices):,} / 10,000")
    
    return (val_indices, val_targets, tunev_indices, tunev_targets, 
            test_indices, test_targets_final)

def compute_class_weights(train_class_counts: List[int]) -> np.ndarray:
    """
    Compute class weights based on train distribution for reweighting.
    
    Args:
        train_class_counts: Number of samples per class in training set
        
    Returns:
        numpy array of weights (normalized to sum to 1)
        
    Example:
        train_counts = [500, 250, 50]  # 3 classes
        weights = [0.625, 0.3125, 0.0625]  # normalized proportions
    """
    train_counts = np.array(train_class_counts)
    total_train = train_counts.sum()
    weights = train_counts / total_train
    
    print("\n=== CLASS WEIGHTS (for reweighting metrics) ===")
    print(f"Total training samples: {total_train:,}")
    print(f"Head class (0): {train_counts[0]} samples, weight={weights[0]:.6f}")
    print(f"Tail class (99): {train_counts[-1]} samples, weight={weights[-1]:.6f}")
    print(f"Weight ratio (head/tail): {weights[0]/weights[-1]:.1f}x")
    
    return weights

def analyze_distribution(indices: List[int], targets: List[int], name: str, 
                        train_counts: Optional[List[int]] = None):
    """Analyze and print distribution statistics."""
    print(f"\n=== {name.upper()} DISTRIBUTION ===")
    
    target_counts = Counter(targets)
    sorted_counts = [target_counts.get(i, 0) for i in range(100)]
    
    total = sum(sorted_counts)
    head_count = sorted_counts[0]
    tail_count = sorted_counts[99]
    
    print(f"Total samples: {total:,}")
    print(f"Head class (0): {head_count} samples ({head_count/total*100:.2f}%)")
    print(f"Tail class (99): {tail_count} samples ({tail_count/total*100:.2f}%)")
    if tail_count > 0:
        print(f"Imbalance factor: {head_count/tail_count:.1f}")
    
    # Group analysis
    groups = {
        'Head (0-9)': sum(sorted_counts[0:10]),
        'Medium (10-49)': sum(sorted_counts[10:50]), 
        'Low (50-89)': sum(sorted_counts[50:90]),
        'Tail (90-99)': sum(sorted_counts[90:100])
    }
    
    print("Distribution by groups:")
    for group_name, group_count in groups.items():
        print(f"  {group_name}: {group_count:,} samples ({group_count/total*100:.1f}%)")

def save_splits_to_json(splits_dict: Dict, output_dir: str):
    """Save all splits to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving splits to {output_dir}...")
    
    for split_name, data in splits_dict.items():
        filepath = output_path / f"{split_name}.json"
        
        # Convert numpy types to Python native types for JSON serialization
        if hasattr(data, 'tolist'):
            data_to_save = data.tolist()
        elif isinstance(data, (list, tuple)):
            data_to_save = [float(x) if isinstance(x, (np.floating, float)) else int(x) 
                           for x in data]
        else:
            data_to_save = list(data)
            
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f)
        
        if isinstance(data_to_save, list) and len(data_to_save) > 0:
            print(f"  Saved {split_name}: {len(data_to_save):,} items")
        else:
            print(f"  Saved {split_name}")

def create_reweighted_cifar100_lt_splits(
    imb_factor: float = 100,
    output_dir: str = "data/cifar100_lt_if100_splits_reweighted", 
    val_ratio: float = 0.15,
    tunev_ratio: float = 0.10,
    seed: int = 42
):
    """
    Create complete CIFAR-100-LT dataset splits with REWEIGHTING approach.
    
    Key differences from duplication approach:
    1. Val/TuneV/Test are all BALANCED (no duplication)
    2. All splits are DISJOINT (no data leakage)
    3. Reweighting happens at METRIC computation (not in data)
    """
    print("=" * 60)
    print("CREATING CIFAR-100-LT DATASET SPLITS (REWEIGHTING)")  
    print("=" * 60)
    
    # Load original CIFAR-100
    print("Loading CIFAR-100 datasets...")
    cifar_train = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=None)
    cifar_test = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=None)
    
    # 1. Create long-tail training set (UNCHANGED)
    train_indices, train_targets, train_counts = create_longtail_train(cifar_train, imb_factor, seed)
    
    # 2. Split balanced test set into Val, TuneV, and Test (NEW - no duplication)
    (val_indices, val_targets, tunev_indices, tunev_targets, 
     test_indices, test_targets) = create_balanced_test_splits(
        cifar_test, val_ratio, tunev_ratio, seed
    )
    
    # 3. Compute class weights for reweighting (NEW)
    class_weights = compute_class_weights(train_counts)
    
    # 4. Analyze all distributions
    analyze_distribution(train_indices, train_targets, "TRAIN")
    analyze_distribution(val_indices, val_targets, "VALIDATION (Balanced)")
    analyze_distribution(tunev_indices, tunev_targets, "TUNEV (Balanced)")
    analyze_distribution(test_indices, test_targets, "TEST (Balanced)")
    
    # 5. Save all splits + class weights + train counts
    splits = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'tunev_indices': tunev_indices,
        'test_indices': test_indices,
        'class_weights': class_weights,  # NEW: for reweighting metrics
        'train_class_counts': train_counts  # NEW: for reference
    }
    
    save_splits_to_json(splits, output_dir)
    
    # 6. Create dataset objects with transforms
    train_transform, eval_transform = get_cifar100_transforms()
    
    datasets = {
        'train': CIFAR100LTDataset(cifar_train, train_indices, train_transform),
        'val': CIFAR100LTDataset(cifar_test, val_indices, eval_transform),
        'tunev': CIFAR100LTDataset(cifar_test, tunev_indices, eval_transform),
        'test': CIFAR100LTDataset(cifar_test, test_indices, eval_transform)
    }
    
    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nIMPORTANT: Val/TuneV/Test are BALANCED.")
    print("Use class_weights.json to reweight metrics during evaluation!")
    
    return datasets, splits, class_weights

if __name__ == "__main__":
    # Create the full dataset
    datasets, splits, class_weights = create_reweighted_cifar100_lt_splits(
        imb_factor=100,
        output_dir="data/cifar100_lt_if100_splits_reweighted",
        val_ratio=0.15,
        tunev_ratio=0.10,
        seed=42
    )
    
    print("\n" + "=" * 60)
    print("DATASETS READY FOR TRAINING")
    print("=" * 60)
    for name, dataset in datasets.items():
        print(f"  {name}: {len(dataset):,} samples")
    
    print("\n" + "=" * 60)
    print("EXAMPLE: How to use class_weights for reweighting")
    print("=" * 60)
    print("""
    # Load class weights
    with open('data/cifar100_lt_if100_splits_reweighted/class_weights.json', 'r') as f:
        class_weights = np.array(json.load(f))
    
    # Compute reweighted accuracy
    per_class_correct = np.array([...])  # correct per class
    per_class_total = np.array([...])    # total per class
    per_class_acc = per_class_correct / per_class_total
    
    # Weighted average (this is the CORRECT metric for long-tail)
    reweighted_acc = (per_class_acc * class_weights).sum()
    """)
