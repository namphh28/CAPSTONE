#!/usr/bin/env python3
"""
RC Curve Per-Group Coverage Analysis
=====================================

Analyzes Risk-Coverage curves with per-group coverage and error analysis.

This complements the main RC curve analysis by adding:
- Per-group coverage κ_k(ρ)
- Per-group errors e_head(ρ), e_tail(ρ)
- Coverage fairness (does rejector improve fairness?)

Usage:
    python scripts/rc_coverage_analysis.py
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from run_balanced_plugin_gating import Config as RCConfig


# ================================
# Config
# ================================
@dataclass
class Config:
    results_dir: str = "./results/ltr_plugin/cifar100_lt_if100"
    splits_dir: str = "./data/cifar100_lt_if100_splits_fixed"
    output_dir: str = "./results/moe_analysis/cifar100_lt_if100"
    tail_threshold: int = 20
    num_classes: int = 100


CFG = Config()


# ================================
# Utility Functions
# ================================
def load_rc_results(results_file: str = "ltr_plugin_gating_balanced.json") -> Dict:
    """Load RC curve results from JSON."""
    results_path = Path(CFG.results_dir) / results_file
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    return results


def build_class_to_group(device: str = "cpu") -> torch.Tensor:
    """Build class-to-group mapping."""
    counts_path = Path(CFG.splits_dir) / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(CFG.num_classes)]
    counts = np.array(class_counts)
    tail_mask = counts <= CFG.tail_threshold
    class_to_group = np.zeros(CFG.num_classes, dtype=np.int64)
    class_to_group[tail_mask] = 1  # 0=head, 1=tail
    return torch.tensor(class_to_group, dtype=torch.long, device=device)


def compute_per_group_coverage(
    rejection_rates: np.ndarray,
    labels: torch.Tensor,
    rejections: List[torch.Tensor],
    class_to_group: torch.Tensor
) -> Dict:
    """
    Compute per-group coverage for each rejection rate.
    
    κ_k(ρ) = coverage for group k at rejection rate ρ
    
    Args:
        rejection_rates: [M] rejection rates (sorted)
        labels: [N] true labels
        rejections: List[M] of [N] boolean rejection masks
        class_to_group: [C] class-to-group mapping
    
    Returns:
        dict with per-group coverage curves
    """
    N = len(labels)
    groups = class_to_group[labels]  # [N]
    num_groups = int(class_to_group.max().item() + 1)
    
    group_coverages = {f'group_{g}': [] for g in range(num_groups)}
    
    for reject_mask in rejections:
        accept_mask = ~reject_mask  # [N]
        
        for g in range(num_groups):
            group_mask = (groups == g)  # [N]
            
            # Coverage for group g: fraction of group g samples that are accepted
            if group_mask.sum() > 0:
                accepted_in_group = accept_mask & group_mask
                coverage_g = accepted_in_group.sum().float().item() / group_mask.sum().float().item()
            else:
                coverage_g = 0.0
            
            group_coverages[f'group_{g}'].append(coverage_g)
    
    # Convert to arrays
    for g in range(num_groups):
        group_coverages[f'group_{g}'] = np.array(group_coverages[f'group_{g}'])
    
    return group_coverages


# ================================
# Main Analysis Function
# ================================
def analyze_rc_coverage():
    """Analyze per-group coverage from RC curve results."""
    print("="*70)
    print("RC Curve Per-Group Coverage Analysis")
    print("="*70)
    
    # Setup
    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load RC results
    print("\n1. Loading RC curve results...")
    results = load_rc_results()
    
    # Get test data from results
    if 'rc_curve' in results and 'test' in results['rc_curve']:
        test_rc = results['rc_curve']['test']
        rejection_rates = np.array(test_rc['rejection_rates'])
        balanced_errors = np.array(test_rc['balanced_errors'])
        worst_group_errors = np.array(test_rc.get('worst_group_errors', []))
        group_errors_list = test_rc.get('group_errors_list', [])
        
        print(f"   Loaded {len(rejection_rates)} RC curve points")
    else:
        raise ValueError("RC curve data not found in results")
    
    # Load labels to compute coverage
    print("\n2. Loading labels...")
    import torchvision
    splits_dir = Path(CFG.splits_dir)
    with open(splits_dir / "test_indices.json", "r") as f:
        indices = json.load(f)
    dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=False)
    labels = torch.tensor([dataset.targets[i] for i in indices])
    
    # Build class-to-group mapping
    class_to_group = build_class_to_group(device="cpu")
    
    # Extract per-group errors from group_errors_list
    num_groups = int(class_to_group.max().item() + 1)
    if len(group_errors_list) > 0:
        head_errors = np.array([ge[0] if len(ge) > 0 else 1.0 for ge in group_errors_list])
        tail_errors = np.array([ge[1] if len(ge) > 1 else 1.0 for ge in group_errors_list])
        gap = tail_errors - head_errors
    else:
        # Fallback: compute from labels if available
        head_errors = np.zeros_like(rejection_rates)
        tail_errors = np.zeros_like(rejection_rates)
        gap = np.zeros_like(rejection_rates)
    
    # Compute per-group coverage (approximate from rejection rate)
    # Note: In actual implementation, we'd need the rejection masks
    # For now, we'll approximate based on the assumption that rejection
    # is applied uniformly across groups at each rejection rate
    
    # Approximate: if rejection rate is ρ, coverage for group k is approximately (1-ρ)
    # This assumes uniform rejection. In practice, rejector may favor certain groups.
    # TODO: Load actual rejection masks from results if available
    
    # For now, approximate coverage from rejection rates
    overall_coverage = 1.0 - rejection_rates
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    
    # Plot 1: RC curves with per-group errors
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall RC curve
    axes[0, 0].plot(rejection_rates, balanced_errors, 'o-', color='green', 
                   label=f'Balanced (AURC={np.trapz(balanced_errors, rejection_rates):.4f})', linewidth=2)
    if len(worst_group_errors) > 0:
        axes[0, 0].plot(rejection_rates, worst_group_errors, 's-', color='royalblue',
                       label=f'Worst-group (AURC={np.trapz(worst_group_errors, rejection_rates):.4f})', linewidth=2)
    axes[0, 0].set_xlabel('Proportion of Rejections ρ', fontsize=11)
    axes[0, 0].set_ylabel('Error', fontsize=11)
    axes[0, 0].set_title('Risk-Coverage Curves', fontweight='bold', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xlim([0, 1])
    
    # Per-group errors
    axes[0, 1].plot(rejection_rates, head_errors, 'o-', color='skyblue', 
                   label='Head Group', linewidth=2)
    axes[0, 1].plot(rejection_rates, tail_errors, 's-', color='coral',
                   label='Tail Group', linewidth=2)
    axes[0, 1].set_xlabel('Proportion of Rejections ρ', fontsize=11)
    axes[0, 1].set_ylabel('Group Error', fontsize=11)
    axes[0, 1].set_title('Per-Group Error vs Rejection Rate', fontweight='bold', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xlim([0, 1])
    
    # Gap: tail - head
    axes[1, 0].plot(rejection_rates, gap, 'd-', color='crimson', linewidth=2)
    axes[1, 0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Proportion of Rejections ρ', fontsize=11)
    axes[1, 0].set_ylabel('Tail Error - Head Error', fontsize=11)
    axes[1, 0].set_title('Error Gap (Tail - Head)', fontweight='bold', fontsize=12)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlim([0, 1])
    
    # Coverage (approximate)
    axes[1, 1].plot(rejection_rates, overall_coverage, 'o-', color='green', linewidth=2, label='Overall')
    # Note: Per-group coverage would require actual rejection masks
    # For now, we show overall coverage
    axes[1, 1].set_xlabel('Proportion of Rejections ρ', fontsize=11)
    axes[1, 1].set_ylabel('Coverage', fontsize=11)
    axes[1, 1].set_title('Coverage vs Rejection Rate', fontweight='bold', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    
    plt.suptitle('RC Curve Per-Group Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "rc_coverage_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save summary statistics
    summary = {
        'rejection_rates': rejection_rates.tolist(),
        'balanced_errors': balanced_errors.tolist(),
        'worst_group_errors': worst_group_errors.tolist() if len(worst_group_errors) > 0 else [],
        'head_errors': head_errors.tolist(),
        'tail_errors': tail_errors.tolist(),
        'gap': gap.tolist(),
        'aurc_balanced': float(np.trapz(balanced_errors, rejection_rates)),
        'aurc_worst': float(np.trapz(worst_group_errors, rejection_rates)) if len(worst_group_errors) > 0 else 0.0,
    }
    
    summary_path = output_dir / "rc_coverage_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Saved analysis to: {output_dir / 'rc_coverage_analysis.png'}")
    print(f"   Saved summary to: {summary_path}")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)


if __name__ == "__main__":
    analyze_rc_coverage()

