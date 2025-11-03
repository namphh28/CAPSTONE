#!/usr/bin/env python3
"""
iNaturalist 2018 Dataset Splits Generator
==========================================

Generate splits for iNaturalist 2018 dataset from train.json and val.json.
Following same methodology as CIFAR-100-LT balanced_test_splits.py

Key Features:
- Dataset already has long-tail distribution (no need to create)
- Uses COCO-style JSON annotation files
- Threshold: 20 samples to distinguish head/tail classes
- Visualization: Last 100 classes only (due to 8000+ classes)
- Val → Test/Val/TuneV: 1:1:1 split (24k samples available)
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
from PIL import Image
import sys
from datetime import datetime


class INaturalistDataset:
    """Wrapper for iNaturalist dataset from JSON files."""
    
    def __init__(self, data_dir: str, json_path: str, transform=None):
        """
        Args:
            data_dir: Directory containing images (e.g., 'train_val2018')
            json_path: Path to JSON annotation file (COCO format)
            transform: Optional transforms
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Parse COCO-style format
        self.images = data['images']  # List of {file_name, id, ...}
        self.annotations = {ann['image_id']: ann['category_id'] for ann in data['annotations']}
        
        # Create mapping from index to (image_path, label)
        self.samples = []
        for img in self.images:
            img_id = img['id']
            file_name = img['file_name']
            image_path = self.data_dir / file_name
            label = self.annotations[img_id]
            self.samples.append((image_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_inaturalist_transforms():
    """Get iNaturalist transforms (ImageNet-style preprocessing)."""
    
    # Training transforms (RandomResizedCrop + RandomHorizontalFlip)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Evaluation transforms (Resize + CenterCrop)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, eval_transform


def analyze_inaturalist_distribution(
    train_dataset,
    val_dataset,
    visualize_last_n: int = 100
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Analyze iNaturalist 2018 distribution from datasets.
    
    Returns:
        Tuple of (train_counts, val_counts, total_counts) dicts
    """
    print("\n" + "="*80)
    print("ANALYZING iNaturalist 2018 DISTRIBUTION")
    print("="*80)
    
    # Count samples per class
    train_counts = Counter()
    val_counts = Counter()
    
    print("\nCounting train samples...")
    for _, label in train_dataset.samples:
        train_counts[label] += 1
    
    print("Counting validation samples...")
    for _, label in val_dataset.samples:
        val_counts[label] += 1
    
    # Combine counts
    all_classes = set(train_counts.keys()) | set(val_counts.keys())
    total_counts = {cls: train_counts.get(cls, 0) + val_counts.get(cls, 0) 
                    for cls in all_classes}
    
    # Statistics
    num_classes = len(all_classes)
    all_sample_counts = list(total_counts.values())
    min_samples = min(all_sample_counts)
    max_samples = max(all_sample_counts)
    mean_samples = np.mean(all_sample_counts)
    median_samples = np.median(all_sample_counts)
    
    # Additional statistics from paper (imbalance ratio, quartiles)
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    sorted_counts = sorted(all_sample_counts)
    q25 = np.percentile(sorted_counts, 25)
    q75 = np.percentile(sorted_counts, 75)
    std_samples = np.std(all_sample_counts)
    
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total classes: {num_classes:,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print(f"Total samples: {len(train_dataset) + len(val_dataset):,}")
    print(f"\nSamples per class:")
    print(f"  Min:          {min_samples}")
    print(f"  Max:          {max_samples:,}")
    print(f"  Mean:         {mean_samples:.1f}")
    print(f"  Median:       {median_samples:.1f}")
    print(f"  Std Dev:      {std_samples:.1f}")
    print(f"  Q25 (25th percentile): {q25:.1f}")
    print(f"  Q75 (75th percentile): {q75:.1f}")
    print(f"  Imbalance Ratio (max/min): {imbalance_ratio:,.1f}x")
    
    # Head/Tail analysis
    threshold = 20
    head_classes = [cls for cls, count in total_counts.items() if count > threshold]
    tail_classes = [cls for cls, count in total_counts.items() if count <= threshold]
    
    print(f"\n{'='*80}")
    print(f"HEAD/TAIL CLASSIFICATION (threshold = {threshold} samples)")
    print(f"{'='*80}")
    print(f"Head classes (> {threshold} samples): {len(head_classes):,} ({100*len(head_classes)/num_classes:.1f}%)")
    print(f"Tail classes (≤ {threshold} samples): {len(tail_classes):,} ({100*len(tail_classes)/num_classes:.1f}%)")
    
    if len(head_classes) > 0:
        head_counts = [total_counts[cls] for cls in head_classes]
        tail_counts = [total_counts[cls] for cls in tail_classes]
        print(f"\nHead class stats:")
        print(f"  Min: {min(head_counts)}, Max: {max(head_counts):,}, Mean: {np.mean(head_counts):.1f}")
        print(f"Tail class stats:")
        print(f"  Min: {min(tail_counts)}, Max: {max(tail_counts)}, Mean: {np.mean(tail_counts):.1f}")
    
    # Visualize last N tail classes (classes with least samples)
    sorted_classes = sorted(total_counts.items(), key=lambda x: x[1])
    # Take first N classes (least samples) for tail visualization
    tail_n_classes = sorted_classes[:visualize_last_n]
    
    print(f"\n{'='*80}")
    print(f"VISUALIZING {visualize_last_n} TAIL CLASSES (least samples)")
    print(f"{'='*80}")
    
    class_ids = [str(cls) for cls, _ in tail_n_classes]
    counts = [count for _, count in tail_n_classes]
    
    plt.figure(figsize=(20, 8))
    bars = plt.bar(range(len(class_ids)), counts, color='steelblue')
    
    # Highlight threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold} samples)')
    
    plt.xlabel('Class ID (sorted by sample count, ascending)', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Distribution of {visualize_last_n} Tail Classes in iNaturalist 2018\n'
              f'(Classes with least samples, sorted ascending)', 
              fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_ids)), class_ids, rotation=90, fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path("data/inaturalist2018_splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_path = output_dir / f"distribution_tail_{visualize_last_n}_classes.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {viz_path}")
    plt.close()
    
    return train_counts, val_counts, total_counts


def visualize_tail_proportion(train_counts: Dict[int, int], threshold: int, output_dir: Path):
    """
    Visualize tail proportion to verify long-tail distribution.
    
    Args:
        train_counts: Dict mapping class_id to train count
        threshold: Threshold for head/tail classification
        output_dir: Directory to save visualization
    """
    print("\n" + "="*80)
    print("COMPUTING TAIL PROPORTION FOR VERIFICATION")
    print("="*80)
    
    # Count head/tail classes and samples
    tail_classes = [cls for cls, count in train_counts.items() if count <= threshold]
    head_classes = [cls for cls, count in train_counts.items() if count > threshold]
    
    num_tail_classes = len(tail_classes)
    num_head_classes = len(head_classes)
    num_total_classes = len(train_counts)
    
    # Calculate samples for head/tail classes
    tail_samples = sum(train_counts[cls] for cls in tail_classes)
    head_samples = sum(train_counts[cls] for cls in head_classes)
    total_samples = sum(train_counts.values())
    
    # Tail proportion: proportion of SAMPLES (not classes) that are in tail classes
    tail_prop = tail_samples / total_samples if total_samples > 0 else 0
    head_prop = head_samples / total_samples if total_samples > 0 else 0
    
    # Class proportions (for reference)
    tail_class_prop = num_tail_classes / num_total_classes
    head_class_prop = num_head_classes / num_total_classes
    
    print(f"Total classes: {num_total_classes:,}")
    print(f"Head classes (> {threshold} samples): {num_head_classes:,} ({head_class_prop*100:.2f}% of classes)")
    print(f"Tail classes (≤ {threshold} samples): {num_tail_classes:,} ({tail_class_prop*100:.2f}% of classes)")
    print(f"\nTotal samples: {total_samples:,}")
    print(f"Head samples: {head_samples:,} ({head_prop*100:.2f}% of samples)")
    print(f"Tail samples: {tail_samples:,} ({tail_prop*100:.2f}% of samples)")
    print(f"\n✓ Tail proportion (samples): {tail_prop:.4f} (Proportion of SAMPLES in tail classes)")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Class distribution by count bins
    sorted_counts = sorted(train_counts.values())
    bins = np.logspace(np.log10(1), np.log10(max(sorted_counts)), 50)
    ax1.hist(sorted_counts, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax1.set_xlabel('Number of Samples per Class', fontsize=11)
    ax1.set_ylabel('Number of Classes', fontsize=11)
    ax1.set_title('Class Distribution (Log Scale)', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    # Plot 2: Head vs Tail pie chart (by SAMPLES, not classes)
    sizes = [head_samples, tail_samples]
    labels = [f'Head ({head_samples:,} samples)', f'Tail ({tail_samples:,} samples)']
    colors = ['lightblue', 'coral']
    explode = (0.05, 0.1)
    ax2.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title(f'Head/Tail Sample Proportions\n(Threshold: {threshold} samples per class)', 
                  fontsize=13, fontweight='bold')
    
    # Plot 3: Cumulative distribution
    sorted_counts_arr = np.array(sorted_counts)
    cumulative = np.arange(1, len(sorted_counts_arr) + 1)
    ax3.plot(sorted_counts_arr, cumulative, linewidth=2, color='steelblue')
    ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold})')
    ax3.axhline(y=num_head_classes, color='green', linestyle=':', linewidth=1.5,
                label=f'Head classes ({num_head_classes:,})')
    ax3.set_xlabel('Number of Samples per Class', fontsize=11)
    ax3.set_ylabel('Cumulative Number of Classes', fontsize=11)
    ax3.set_title('Cumulative Class Distribution', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Samples per class (sorted, for overview)
    ax4.bar(range(len(sorted_counts)), sorted_counts, color='steelblue', alpha=0.7)
    ax4.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold})')
    ax4.set_xlabel('Class Index (sorted by sample count)', fontsize=11)
    ax4.set_ylabel('Number of Samples', fontsize=11)
    ax4.set_title('All Classes Distribution (Sorted)', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = output_dir / "tail_proportion_analysis.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved tail proportion analysis to: {viz_path}")
    plt.close()
    
    return tail_prop


def split_train_for_expert_and_gating(
    train_dataset,
    train_counts: Dict[int, int],
    expert_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
    """
    Split train set into expert (90%) and gating (10%).
    CRITICAL: Maintains same imbalance ratio!
    
    Args:
        train_dataset: INaturalistDataset instance
        train_counts: Dict mapping class_id to train count
        expert_ratio: Ratio for expert training (default: 0.9)
        seed: Random seed
        
    Returns:
        (expert_indices, expert_targets, expert_counts,
         gating_indices, gating_targets, gating_counts)
    """
    print(f"\n{'='*80}")
    print(f"SPLITTING TRAIN SET ({expert_ratio*100:.0f}% Expert / {(1-expert_ratio)*100:.0f}% Gating)")
    print(f"{'='*80}")
    
    np.random.seed(seed)
    
    # Group indices by class - use samples directly (no image loading needed)
    print("  Grouping samples by class...")
    indices_by_class = defaultdict(list)
    for idx, (_, label) in enumerate(train_dataset.samples):
        indices_by_class[label].append(idx)
    print(f"  ✓ Grouped {len(indices_by_class)} classes")
    
    expert_indices = []
    gating_indices = []
    expert_counts = defaultdict(int)
    gating_counts = defaultdict(int)
    
    # Split each class
    print("  Splitting each class...")
    max_class = max(train_counts.keys()) if train_counts else 0
    classes_to_process = sorted([cls for cls in indices_by_class.keys() if cls <= max_class])
    print(f"  Processing {len(classes_to_process)} classes...")
    
    for i, cls in enumerate(classes_to_process):
        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i+1}/{len(classes_to_process)} classes processed...")
            
        cls_indices = np.array(indices_by_class[cls])
        total_cls = len(cls_indices)
        
        # Calculate split sizes
        expert_size = int(total_cls * expert_ratio)
        gating_size = total_cls - expert_size
        
        # Shuffle and split
        np.random.shuffle(cls_indices)
        expert_cls_indices = cls_indices[:expert_size].tolist()
        gating_cls_indices = cls_indices[expert_size:].tolist()
        
        expert_indices.extend(expert_cls_indices)
        gating_indices.extend(gating_cls_indices)
        expert_counts[cls] = expert_size
        gating_counts[cls] = gating_size
    
    print("  ✓ Completed splitting all classes")
    
    # Get targets directly from samples (no image loading needed)
    print("  Extracting labels from samples...")
    expert_targets = [train_dataset.samples[idx][1] for idx in expert_indices]
    gating_targets = [train_dataset.samples[idx][1] for idx in gating_indices]
    print("  ✓ Extracted all labels")
    
    # Convert counts to lists
    num_classes = max(expert_counts.keys()) + 1 if expert_counts else 0
    expert_counts_list = [expert_counts.get(i, 0) for i in range(num_classes)]
    gating_counts_list = [gating_counts.get(i, 0) for i in range(num_classes)]
    
    # Verify splits
    print(f"\n  SUCCESS: Expert split:")
    print(f"    Total: {len(expert_indices):,} samples")
    if len(expert_counts) > 0:
        non_zero_expert = [c for c in expert_counts_list if c > 0]
        if non_zero_expert:
            print(f"    Mean samples/class: {np.mean(non_zero_expert):.1f}")
            print(f"    Max samples/class: {max(non_zero_expert):,}")
            print(f"    Classes with samples: {len(non_zero_expert):,}")
    
    print(f"\n  SUCCESS: Gating split:")
    print(f"    Total: {len(gating_indices):,} samples")
    if len(gating_counts) > 0:
        non_zero_gating = [c for c in gating_counts_list if c > 0]
        if non_zero_gating:
            print(f"    Mean samples/class: {np.mean(non_zero_gating):.1f}")
            print(f"    Max samples/class: {max(non_zero_gating):,}")
            print(f"    Classes with samples: {len(non_zero_gating):,}")
    
    # Verify no overlap
    expert_set = set(expert_indices)
    gating_set = set(gating_indices)
    assert len(expert_set & gating_set) == 0, "Expert and Gating splits overlap!"
    print("\n  ✓ No overlap between expert and gating splits")
    
    return (expert_indices, expert_targets, expert_counts_list,
            gating_indices, gating_targets, gating_counts_list)


def split_val_into_val_test_tunev(
    val_dataset,
    val_counts: Dict[int, int],
    seed: int = 42
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
    """
    Split validation set into Val/Test/TuneV with 1:1:1 ratio.
    
    Args:
        val_dataset: INaturalistDataset instance
        val_counts: Dict mapping class_id to val count
        seed: Random seed
        
    Returns:
        (val_indices, val_targets, test_indices, test_targets, tunev_indices, tunev_targets)
    """
    print(f"\n{'='*80}")
    print("SPLITTING VALIDATION SET INTO VAL/TEST/TUNEV (1:1:1)")
    print(f"{'='*80}")
    
    np.random.seed(seed)
    
    # Group indices by class - use samples directly (no image loading needed)
    print("  Grouping samples by class...")
    indices_by_class = defaultdict(list)
    for idx, (_, label) in enumerate(val_dataset.samples):
        indices_by_class[label].append(idx)
    print(f"  ✓ Grouped {len(indices_by_class)} classes")
    
    val_indices = []
    test_indices = []
    tunev_indices = []
    
    # Split each class with 1:1:1 ratio
    print("  Splitting each class (1:1:1)...")
    max_class = max(val_counts.keys()) if val_counts else 0
    classes_to_process = sorted([cls for cls in indices_by_class.keys() if cls <= max_class])
    print(f"  Processing {len(classes_to_process)} classes...")
    
    for i, cls in enumerate(classes_to_process):
        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i+1}/{len(classes_to_process)} classes processed...")
            
        cls_indices = np.array(indices_by_class[cls])
        total_cls = len(cls_indices)
        
        # Calculate split sizes (1:1:1)
        split_size = total_cls // 3
        val_size = split_size
        test_size = split_size
        tunev_size = total_cls - val_size - test_size
        
        # Shuffle and split
        np.random.shuffle(cls_indices)
        cls_val_indices = cls_indices[:val_size].tolist()
        cls_test_indices = cls_indices[val_size:val_size+test_size].tolist()
        cls_tunev_indices = cls_indices[val_size+test_size:].tolist()
        
        val_indices.extend(cls_val_indices)
        test_indices.extend(cls_test_indices)
        tunev_indices.extend(cls_tunev_indices)
    
    print("  ✓ Completed splitting all classes")
    
    # Get targets directly from samples (no image loading needed)
    print("  Extracting labels from samples...")
    val_targets = [val_dataset.samples[idx][1] for idx in val_indices]
    test_targets = [val_dataset.samples[idx][1] for idx in test_indices]
    tunev_targets = [val_dataset.samples[idx][1] for idx in tunev_indices]
    print("  ✓ Extracted all labels")
    
    # Verify splits
    print(f"\n  SUCCESS: Splits created:")
    print(f"    Val:   {len(val_indices):,} samples")
    print(f"    Test:  {len(test_indices):,} samples")
    print(f"    TuneV: {len(tunev_indices):,} samples")
    
    # Verify no overlap
    val_set = set(val_indices)
    test_set = set(test_indices)
    tunev_set = set(tunev_indices)
    
    assert len(val_set & test_set) == 0, "Val and Test overlap!"
    assert len(val_set & tunev_set) == 0, "Val and TuneV overlap!"
    assert len(test_set & tunev_set) == 0, "Test and TuneV overlap!"
    print("  ✓ No data leakage - all splits are disjoint")
    
    return (val_indices, val_targets, test_indices, test_targets, tunev_indices, tunev_targets)


def compute_class_weights(train_counts: Dict[int, int]) -> np.ndarray:
    """
    Compute class weights for metric reweighting.
    
    Args:
        train_counts: Dict mapping class_id to train count
        
    Returns:
        numpy array of weights (normalized to sum to 1)
    """
    counts = np.array(list(train_counts.values()))
    total_train = counts.sum()
    weights = counts / total_train
    
    print(f"\n{'='*60}")
    print("CLASS WEIGHTS (for metric reweighting)")
    print(f"{'='*60}")
    print(f"Total training samples: {total_train:,}")
    print(f"Total classes: {len(counts):,}")
    print(f"\nWeight distribution:")
    print(f"  Min:  {weights.min():.6f} (class with {counts[weights.argmin()]} samples)")
    print(f"  Max:  {weights.max():.6f} (class with {counts[weights.argmax()]:,} samples)")
    print(f"  Mean: {weights.mean():.6f}")
    print(f"  Median: {np.median(weights):.6f}")
    if weights.max() > 0 and weights.min() > 0:
        print(f"\nWeight ratio (max/min): {weights.max()/weights.min():.1f}x")
    print(f"{'='*60}")
    
    return weights


def generate_class_distribution_report(
    train_counts: List[int],
    expert_counts: List[int],
    gating_counts: List[int],
    output_dir: Path,
    threshold: int = 20
):
    """
    Generate detailed class distribution report across all ~8000 classes.
    Creates both Markdown and CSV reports showing exact sample counts.
    
    Args:
        train_counts: Original train counts per class
        expert_counts: Expert split counts per class
        gating_counts: Gating split counts per class
        output_dir: Directory to save reports
        threshold: Threshold for head/tail classification
    """
    print(f"\n{'='*80}")
    print("GENERATING CLASS DISTRIBUTION REPORT")
    print(f"{'='*80}")
    
    num_classes = len(train_counts)
    
    # Classify head/tail
    head_mask = np.array(train_counts) > threshold
    tail_mask = ~head_mask
    head_classes = np.where(head_mask)[0]
    tail_classes = np.where(tail_mask)[0]
    
    print(f"Total classes: {num_classes:,}")
    print(f"Head classes (> {threshold} samples): {len(head_classes):,}")
    print(f"Tail classes (≤ {threshold} samples): {len(tail_classes):,}")
    
    # Compute statistics
    total_train = sum(train_counts)
    total_expert = sum(expert_counts)
    total_gating = sum(gating_counts)
    
    # Create comprehensive data
    class_data = []
    for class_id in range(num_classes):
        train_count = train_counts[class_id]
        expert_count = expert_counts[class_id]
        gating_count = gating_counts[class_id]
        
        # Calculate ratios
        train_pct = (train_count / total_train * 100) if total_train > 0 else 0
        expert_pct = (expert_count / total_expert * 100) if total_expert > 0 else 0
        gating_pct = (gating_count / total_gating * 100) if total_gating > 0 else 0
        
        # Verify split (should be ~90% expert, ~10% gating)
        split_ratio = (expert_count / train_count) if train_count > 0 else 0
        
        # Category
        category = "Head" if train_count > threshold else "Tail"
        
        class_data.append({
            'class_id': class_id,
            'category': category,
            'train_samples': train_count,
            'expert_samples': expert_count,
            'gating_samples': gating_count,
            'train_pct': train_pct,
            'expert_pct': expert_pct,
            'gating_pct': gating_pct,
            'split_ratio': split_ratio,
        })
    
    # Sort by train count (descending)
    class_data.sort(key=lambda x: x['train_samples'], reverse=True)
    
    # Generate Markdown report
    md_path = output_dir / "class_distribution_report.md"
    print(f"\nGenerating Markdown report: {md_path}")
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# iNaturalist 2018 Class Distribution Report\n\n")
        f.write("**Generated:** " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write("---\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Classes:** {num_classes:,}\n")
        f.write(f"- **Head Classes (> {threshold} samples):** {len(head_classes):,} ({len(head_classes)/num_classes*100:.1f}%)\n")
        f.write(f"- **Tail Classes (≤ {threshold} samples):** {len(tail_classes):,} ({len(tail_classes)/num_classes*100:.1f}%)\n")
        f.write(f"\n- **Total Train Samples:** {total_train:,}\n")
        f.write(f"- **Total Expert Samples:** {total_expert:,} ({total_expert/total_train*100:.2f}%)\n")
        f.write(f"- **Total Gating Samples:** {total_gating:,} ({total_gating/total_train*100:.2f}%)\n\n")
        
        # Head vs Tail stats
        head_train = sum(train_counts[i] for i in head_classes)
        tail_train = sum(train_counts[i] for i in tail_classes)
        f.write("- **Head Samples:** {:,} ({:.2f}%)\n".format(head_train, head_train/total_train*100))
        f.write("- **Tail Samples:** {:,} ({:.2f}%)\n\n".format(tail_train, tail_train/total_train*100))
        f.write("---\n\n")
        
        # Detailed table
        f.write("## Detailed Class Distribution\n\n")
        f.write("| Class ID | Category | Train | Expert | Gating | Train % | Expert % | Gating % | Split Ratio |\n")
        f.write("|----------|----------|-------|--------|--------|---------|----------|----------|-------------|\n")
        
        # Show all classes
        for cls in class_data:
            f.write("| {class_id} | {category} | {train_samples:,} | {expert_samples} | {gating_samples} | "
                   "{train_pct:.3f}% | {expert_pct:.3f}% | {gating_pct:.3f}% | {split_ratio:.2f} |\n".format(**cls))
    
    print(f"✓ Saved Markdown report: {md_path}")
    
    # Generate CSV report for easier analysis
    csv_path = output_dir / "class_distribution_report.csv"
    print(f"Generating CSV report: {csv_path}")
    
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['class_id', 'category', 'train_samples', 'expert_samples', 'gating_samples',
                     'train_pct', 'expert_pct', 'gating_pct', 'split_ratio']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(class_data)
    
    print(f"✓ Saved CSV report: {csv_path}")
    
    # Also create a compact summary by bins
    summary_path = output_dir / "class_distribution_summary.md"
    print(f"Generating summary report: {summary_path}")
    
    # Bin classes by sample count ranges
    bins = [
        (0, 5, "0-5"),
        (6, 10, "6-10"),
        (11, 20, "11-20"),
        (21, 50, "21-50"),
        (51, 100, "51-100"),
        (101, 200, "101-200"),
        (201, 500, "201-500"),
        (501, 1000, "501-1000"),
        (1001, 5000, "1001-5000"),
        (5001, float('inf'), "5000+"),
    ]
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# iNaturalist 2018 Class Distribution Summary\n\n")
        f.write("**Generated:** " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write("---\n\n")
        
        f.write("## Distribution by Sample Count Ranges\n\n")
        f.write("| Range | # Classes | % Classes | Train Samples | % Train | Expert Samples | Gating Samples |\n")
        f.write("|-------|-----------|-----------|---------------|---------|----------------|----------------|\n")
        
        for min_val, max_val, label in bins:
            mask = np.array([min_val <= cnt <= max_val for cnt in train_counts])
            num_classes_in_bin = mask.sum()
            pct_classes = (num_classes_in_bin / num_classes * 100) if num_classes > 0 else 0
            
            bin_train_samples = sum(train_counts[i] for i in range(num_classes) if mask[i])
            bin_expert_samples = sum(expert_counts[i] for i in range(num_classes) if mask[i])
            bin_gating_samples = sum(gating_counts[i] for i in range(num_classes) if mask[i])
            
            bin_train_pct = (bin_train_samples / total_train * 100) if total_train > 0 else 0
            
            f.write(f"| {label} | {num_classes_in_bin:,} | {pct_classes:.1f}% | "
                   f"{bin_train_samples:,} | {bin_train_pct:.2f}% | "
                   f"{bin_expert_samples:,} | {bin_gating_samples:,} |\n")
        
        # Special row for head vs tail
        f.write("\n| Category | # Classes | % Classes | Train Samples | % Train | Expert Samples | Gating Samples |\n")
        f.write("|----------|-----------|-----------|---------------|---------|----------------|----------------|\n")
        
        head_train_samples = sum(train_counts[i] for i in head_classes)
        head_expert_samples = sum(expert_counts[i] for i in head_classes)
        head_gating_samples = sum(gating_counts[i] for i in head_classes)
        
        tail_train_samples = sum(train_counts[i] for i in tail_classes)
        tail_expert_samples = sum(expert_counts[i] for i in tail_classes)
        tail_gating_samples = sum(gating_counts[i] for i in tail_classes)
        
        f.write(f"| Head (> {threshold}) | {len(head_classes):,} | {len(head_classes)/num_classes*100:.1f}% | "
               f"{head_train_samples:,} | {head_train_samples/total_train*100:.2f}% | "
               f"{head_expert_samples:,} | {head_gating_samples:,} |\n")
        
        f.write(f"| Tail (≤ {threshold}) | {len(tail_classes):,} | {len(tail_classes)/num_classes*100:.1f}% | "
               f"{tail_train_samples:,} | {tail_train_samples/total_train*100:.2f}% | "
               f"{tail_expert_samples:,} | {tail_gating_samples:,} |\n")
    
    print(f"✓ Saved summary report: {summary_path}")
    print(f"\n{'='*80}")
    print("CLASS DISTRIBUTION REPORTS GENERATED!")
    print(f"{'='*80}")


def save_splits_to_json(splits_dict: Dict, output_dir: str):
    """Save all splits and metadata to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving splits to: {output_dir}")
    
    for split_name, data in splits_dict.items():
        filepath = output_path / f"{split_name}.json"
        
        # Convert numpy types to Python native types
        if hasattr(data, 'tolist'):
            data_to_save = data.tolist()
        elif isinstance(data, (list, tuple)):
            data_to_save = [float(x) if isinstance(x, (np.floating, float)) else int(x) 
                           for x in data]
        else:
            data_to_save = data
            
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        if isinstance(data_to_save, (list, dict)) and len(data_to_save) > 0:
            size = len(data_to_save)
            print(f"  - {split_name}.json ({size:,} items)")
        else:
            print(f"  - {split_name}.json")


def visualize_expert_gating_split(
    train_counts: List[int],
    expert_counts: List[int],
    gating_counts: List[int],
    output_dir: Path,
    num_classes_to_show: int = 100
):
    """
    Visualize Expert/Gating split for tail classes (classes with least samples).
    
    Args:
        train_counts: Original train counts per class
        expert_counts: Expert split counts per class
        gating_counts: Gating split counts per class
        output_dir: Directory to save visualization
        num_classes_to_show: Number of tail classes to visualize (least samples)
    """
    # Sort by train count (ascending) to show tail classes (least samples first)
    sorted_idx = np.argsort(train_counts)
    # Take first N indices (tail classes with least samples)
    tail_n_idx = sorted_idx[:num_classes_to_show]
    
    # Extract counts for visualization
    expert_show = [expert_counts[i] for i in tail_n_idx]
    gating_show = [gating_counts[i] for i in tail_n_idx]
    class_ids_show = [str(i) for i in tail_n_idx]
    
    # Create visualization with 2 separate plots for Expert and Gating
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    
    # Plot 1: Expert split distribution (90% từ train set)
    ax1.bar(range(len(class_ids_show)), expert_show, color='forestgreen', alpha=0.8, label='Expert (90% of train)')
    ax1.axhline(y=20*0.9, color='darkgreen', linestyle='--', linewidth=2, 
                label=f'Expected threshold ({20*0.9:.0f} samples)')
    ax1.set_xlabel('Class ID (sorted by sample count, ascending - tail classes)', fontsize=11)
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.set_title(f'Expert Split: 90% of Train Data ({num_classes_to_show} Tail Classes - Least Samples)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(class_ids_show)))
    ax1.set_xticklabels(class_ids_show, rotation=90, fontsize=7)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Gating split distribution (10% từ train set)
    ax2.bar(range(len(class_ids_show)), gating_show, color='coral', alpha=0.8, label='Gating (10% of train)')
    ax2.axhline(y=20*0.1, color='darkred', linestyle='--', linewidth=2,
                label=f'Expected threshold ({20*0.1:.0f} samples)')
    ax2.set_xlabel('Class ID (sorted by sample count, ascending - tail classes)', fontsize=11)
    ax2.set_ylabel('Number of Samples', fontsize=11)
    ax2.set_title(f'Gating Split: 10% of Train Data ({num_classes_to_show} Tail Classes - Least Samples)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(class_ids_show)))
    ax2.set_xticklabels(class_ids_show, rotation=90, fontsize=7)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save
    viz_path = output_dir / f"expert_gating_split_tail_{num_classes_to_show}_classes.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved Expert/Gating visualization to: {viz_path}")
    plt.close()


def create_inaturalist2018_splits(
    train_json_path: str,
    val_json_path: str,
    data_dir: str = "data/inaturalist2018",
    output_dir: str = "data/inaturalist2018_splits",
    seed: int = 42,
    expert_ratio: float = 0.9,
    visualize_last_n: int = 100,
    log_file: Optional[str] = None
):
    """
    Create iNaturalist 2018 dataset splits from JSON files.
    
    Args:
        train_json_path: Path to train.json (COCO format)
        val_json_path: Path to val.json (COCO format)
        data_dir: Directory containing images
        output_dir: Output directory for splits JSON files
        seed: Random seed
        expert_ratio: Ratio for expert training
        visualize_last_n: Number of last classes to visualize
        log_file: Optional path to log file. If provided, all output will be logged.
        
    Returns:
        Tuple of (splits_dict, class_weights)
    """
    # Setup logging if log_file is provided
    original_stdout = sys.stdout
    log_file_handle = None
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_handle = open(log_path, 'w', encoding='utf-8')
        
        # Create a class that writes to both stdout and log file
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            
            def flush(self):
                for f in self.files:
                    f.flush()
        
        sys.stdout = TeeOutput(original_stdout, log_file_handle)
        print(f"\n{'='*80}")
        print(f"LOGGING TO FILE: {log_path}")
        print(f"STARTED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
    
    try:
        print("="*80)
        print("CREATING iNaturalist 2018 DATASET SPLITS")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Train JSON: {train_json_path}")
        print(f"  Val JSON: {val_json_path}")
        print(f"  Data Directory: {data_dir}")
        print(f"  Output Directory: {output_dir}")
        print(f"  Random Seed: {seed}")
        print(f"  Train Split: {expert_ratio*100:.0f}% Expert / {(1-expert_ratio)*100:.0f}% Gating")
        
        # Load datasets
        print("\n" + "="*80)
        print("STEP 1: Loading iNaturalist 2018 datasets from JSON...")
        print("="*80)
        
        train_dataset = INaturalistDataset(data_dir, train_json_path)
        val_dataset = INaturalistDataset(data_dir, val_json_path)
        
        print(f"  - Train: {len(train_dataset):,} samples")
        print(f"  - Val:   {len(val_dataset):,} samples")
        
        # Analyze distribution
        print("\n" + "="*80)
        print("STEP 2: Analyzing distribution...")
        print("="*80)
        train_counts, val_counts, total_counts = analyze_inaturalist_distribution(
            train_dataset, val_dataset, visualize_last_n=visualize_last_n
        )
        
        # Convert counts to lists
        max_class = max(total_counts.keys())
        train_counts_list = [train_counts.get(i, 0) for i in range(max_class + 1)]
        val_counts_list = [val_counts.get(i, 0) for i in range(max_class + 1)]
        total_counts_list = [total_counts.get(i, 0) for i in range(max_class + 1)]
        
        # Visualize tail proportion to verify long-tail distribution
        tail_threshold = 20
        tail_prop = visualize_tail_proportion(train_counts, tail_threshold, Path(output_dir))
        
        # Split train into expert and gating
        print("\n" + "="*80)
        print("STEP 3: Splitting train set (Expert vs Gating)...")
        print("="*80)
        (expert_indices, expert_targets, expert_counts,
         gating_indices, gating_targets, gating_counts) = split_train_for_expert_and_gating(
            train_dataset, train_counts, expert_ratio, seed
        )
        
        # Visualize Expert/Gating split
        visualize_expert_gating_split(
            train_counts_list, expert_counts, gating_counts,
            Path(output_dir), num_classes_to_show=visualize_last_n
        )
        
        # Split validation into Val/Test/TuneV
        print("\n" + "="*80)
        print("STEP 4: Splitting validation set (1:1:1)...")
        print("="*80)
        (val_indices, val_targets,
         test_indices, test_targets,
         tunev_indices, tunev_targets) = split_val_into_val_test_tunev(
            val_dataset, val_counts, seed
        )
        
        # Compute class weights
        print("\n" + "="*80)
        print("STEP 5: Computing class weights...")
        print("="*80)
        class_weights = compute_class_weights(train_counts)
        
        # Save all splits
        print("\n" + "="*80)
        print("STEP 6: Saving splits...")
        print("="*80)
        
        splits = {
            'train_class_counts': train_counts_list,
            'val_class_counts': val_counts_list,
            'total_class_counts': total_counts_list,
            'expert_indices': expert_indices,
            'expert_targets': expert_targets,
            'expert_class_counts': expert_counts,
            'gating_indices': gating_indices,
            'gating_targets': gating_targets,
            'gating_class_counts': gating_counts,
            'val_indices': val_indices,
            'val_targets': val_targets,
            'test_indices': test_indices,
            'test_targets': test_targets,
            'tunev_indices': tunev_indices,
            'tunev_targets': tunev_targets,
            'class_weights': class_weights.tolist(),
        }
        
        save_splits_to_json(splits, output_dir)
        
        # Generate detailed class distribution report
        print("\n" + "="*80)
        print("STEP 7: Generating class distribution reports...")
        print("="*80)
        generate_class_distribution_report(
            train_counts_list, expert_counts, gating_counts,
            Path(output_dir), threshold=tail_threshold
        )
        
        # Final summary
        print("\n" + "="*80)
        print("SUCCESS: DATASET CREATION COMPLETED!")
        print("="*80)
        print("\nSummary:")
        print(f"  Total classes: {max_class + 1:,}")
        print(f"  Tail proportion (samples): {tail_prop:.4f} ({tail_prop*100:.2f}% of samples are in tail classes)")
        print(f"  Expert split: {len(expert_indices):,} samples")
        print(f"  Gating split: {len(gating_indices):,} samples")
        print(f"  Val split:   {len(val_indices):,} samples")
        print(f"  Test split:  {len(test_indices):,} samples")
        print(f"  TuneV split: {len(tunev_indices):,} samples")
        
        print("\nVisualizations saved:")
        print(f"  - distribution_tail_{visualize_last_n}_classes.png")
        print(f"  - expert_gating_split_tail_{visualize_last_n}_classes.png")
        print(f"  - tail_proportion_analysis.png")
        
        print("\nReports saved:")
        print(f"  - class_distribution_report.md (detailed table)")
        print(f"  - class_distribution_report.csv (CSV format)")
        print(f"  - class_distribution_summary.md (summary by bins)")
        
        print("\nNext steps:")
        print("  1. Train experts with expert split (ResNet-50, batch=1024, cosine scheduler)")
        print("  2. Train gating with gating split")
        print("  3. Evaluate with reweighted metrics")
        
        if log_file:
            print(f"\n{'='*80}")
            print(f"COMPLETED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"LOG FILE SAVED: {log_file}")
            print(f"{'='*80}")
        
        return splits, class_weights
    
    finally:
        # Restore stdout and close log file
        if log_file_handle:
            sys.stdout = original_stdout
            log_file_handle.close()


if __name__ == "__main__":
    # Example usage - adjust paths as needed
    import sys
    
    # You can run this directly or import it
    print("="*80)
    print("iNaturalist 2018 Splits Generator")
    print("="*80)
    print("\nPlease use the script: scripts/create_inaturalist_splits.py")
    print("Example:")
    print("  python scripts/create_inaturalist_splits.py \\")
    print("    --train-json /path/to/train.json \\")
    print("    --val-json /path/to/val.json \\")
    print("    --data-dir data/inaturalist2018/train_val2018")
    print("="*80)

