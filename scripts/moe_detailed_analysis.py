#!/usr/bin/env python3
"""
Detailed MoE-L2R Analysis - Extended Plots
==========================================

Creates detailed plots to demonstrate why MoE helps the rejector:

1. Probability level (why MoE helps):
   - A. Reliability & NLL decomposition
   - B. Margin & ranking stability
   - C. Disagreement & entropy

2. Group level:
   - D. Per-group RC curves & AURC
   - E. Fairness by coverage (α-coverage per group)
   - F. Who gets rejected (rejection heatmap)

3. MoE mechanism:
   - G. Gating weight distribution by group
   - H. Variance decomposition & MI
   - I. Counterfactuals (Oracle/Uniform/Gating RC, shuffle-gating, ablation)

4. Connection to L2R theory:
   - J. Plugin formula validation

Usage:
    python scripts/moe_detailed_analysis.py
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial.distance import jaccard

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.moe_l2r_comprehensive_analysis import (
    Config, load_expert_logits, load_labels, load_gating_network,
    build_class_to_group, compute_mixture_posterior, convert_to_json_serializable
)


CFG = Config()


# ================================
# A. Reliability & NLL Decomposition
# ================================
def plot_nll_decomposition(expert_posteriors: torch.Tensor, uniform_mixture: torch.Tensor,
                           gating_mixture: torch.Tensor, labels: torch.Tensor,
                           expert_names: List[str], save_path: Path):
    """Plot NLL(mixture) vs mean NLL(experts)."""
    N, E, C = expert_posteriors.shape
    eps = 1e-8
    
    # Compute NLL for each expert
    expert_nlls = []
    for e in range(E):
        true_probs = expert_posteriors[torch.arange(N), e, labels]
        nll_e = -torch.log(true_probs + eps).mean().item()
        expert_nlls.append(nll_e)
    
    mean_expert_nll = np.mean(expert_nlls)
    
    # Compute NLL for mixtures
    uniform_true_probs = uniform_mixture[torch.arange(N), labels]
    uniform_nll = -torch.log(uniform_true_probs + eps).mean().item()
    
    gating_true_probs = gating_mixture[torch.arange(N), labels]
    gating_nll = -torch.log(gating_true_probs + eps).mean().item()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Point-point plot: NLL(mix) vs mean NLL(experts)
    methods = ['Uniform-Mix', 'Gating-Mix']
    nll_mixtures = [uniform_nll, gating_nll]
    
    axes[0].scatter([mean_expert_nll] * 2, nll_mixtures, s=200, alpha=0.7,
                   c=['steelblue', 'green'], edgecolors='black', linewidth=2)
    axes[0].plot([mean_expert_nll, mean_expert_nll], [mean_expert_nll * 0.9, mean_expert_nll * 1.1],
                'k--', linewidth=1, alpha=0.5, label='y=x (equal)')
    axes[0].set_xlabel('Mean NLL(Experts)', fontsize=12)
    axes[0].set_ylabel('NLL(Mixture)', fontsize=12)
    axes[0].set_title('NLL: Mixture vs Mean Experts', fontweight='bold', fontsize=13)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    for i, (method, nll) in enumerate(zip(methods, nll_mixtures)):
        improvement = mean_expert_nll - nll
        axes[0].annotate(f'{method}\n({improvement:+.4f})',
                        xy=(mean_expert_nll, nll), xytext=(10, 10),
                        textcoords='offset points', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Bar plot comparison
    all_methods = expert_names + ['Mean Experts', 'Uniform-Mix', 'Gating-Mix']
    all_nlls = expert_nlls + [mean_expert_nll, uniform_nll, gating_nll]
    colors = ['steelblue'] * E + ['orange', 'steelblue', 'green']
    
    axes[1].bar(range(len(all_methods)), all_nlls, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(all_methods)))
    axes[1].set_xticklabels(all_methods, rotation=15, ha='right')
    axes[1].set_ylabel('NLL', fontsize=12)
    axes[1].set_title('NLL Comparison', fontweight='bold', fontsize=13)
    axes[1].grid(alpha=0.3, axis='y')
    
    for i, nll in enumerate(all_nlls):
        axes[1].text(i, nll + 0.01, f'{nll:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('NLL Decomposition: Mixture vs Experts', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration_vs_coverage(posteriors: torch.Tensor, labels: torch.Tensor,
                                 method_name: str, save_path: Path, n_bins: int = 15):
    """Plot cumulative ECE vs coverage (samples sorted by confidence)."""
    N, C = posteriors.shape
    confidences, predictions = posteriors.max(dim=-1)
    accuracies = (predictions == labels).float()
    
    # Sort by confidence (ascending)
    sorted_indices = torch.argsort(confidences)
    sorted_conf = confidences[sorted_indices].cpu().numpy()
    sorted_acc = accuracies[sorted_indices].cpu().numpy()
    
    # Compute cumulative ECE at different coverage levels
    coverage_levels = np.linspace(0.1, 1.0, 20)
    cumulative_eces = []
    
    for cov in coverage_levels:
        n_accept = int(N * cov)
        if n_accept == 0:
            cumulative_eces.append(1.0)
            continue
        
        accepted_conf = sorted_conf[-n_accept:]
        accepted_acc = sorted_acc[-n_accept:]
        
        # Compute ECE on accepted samples
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bin_mask = (accepted_conf > bin_boundaries[i]) & (accepted_conf <= bin_boundaries[i+1])
            if bin_mask.sum() > 0:
                acc_in_bin = accepted_acc[bin_mask].mean()
                conf_in_bin = accepted_conf[bin_mask].mean()
                prop_in_bin = bin_mask.sum() / len(accepted_conf)
                ece += abs(conf_in_bin - acc_in_bin) * prop_in_bin
        
        cumulative_eces.append(ece)
    
    plt.figure(figsize=(8, 6))
    plt.plot(coverage_levels, cumulative_eces, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Coverage (samples sorted by confidence)', fontsize=12)
    plt.ylabel('Cumulative ECE', fontsize=12)
    plt.title(f'Calibration Error vs Coverage: {method_name}', fontweight='bold', fontsize=13)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ================================
# B. Margin & Ranking Stability
# ================================
def compute_margin(posteriors: torch.Tensor) -> torch.Tensor:
    """Compute margin = max_p - second_max_p."""
    top2_vals, _ = torch.topk(posteriors, k=2, dim=-1)
    margin = top2_vals[:, 0] - top2_vals[:, 1]
    return margin


def plot_margin_analysis(expert_posteriors: torch.Tensor, uniform_mixture: torch.Tensor,
                        gating_mixture: torch.Tensor, labels: torch.Tensor,
                        expert_names: List[str], save_path: Path):
    """Plot margin histograms and AUROC."""
    N, E, C = expert_posteriors.shape
    
    # Compute margins for each method
    margins = {}
    for e in range(E):
        margins[expert_names[e]] = compute_margin(expert_posteriors[:, e, :]).cpu().numpy()
    margins['Uniform-Mix'] = compute_margin(uniform_mixture).cpu().numpy()
    margins['Gating-Mix'] = compute_margin(gating_mixture).cpu().numpy()
    
    # Compute correctness
    is_correct = {}
    for e in range(E):
        preds = expert_posteriors[:, e, :].argmax(dim=-1)
        is_correct[expert_names[e]] = (preds == labels).cpu().numpy()
    
    uniform_preds = uniform_mixture.argmax(dim=-1)
    is_correct['Uniform-Mix'] = (uniform_preds == labels).cpu().numpy()
    
    gating_preds = gating_mixture.argmax(dim=-1)
    is_correct['Gating-Mix'] = (gating_preds == labels).cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram of margins
    ax = axes[0, 0]
    for name, margin in margins.items():
        ax.hist(margin, bins=50, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Margin (max_p - second_max_p)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Margin Distribution', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # KDE of margins
    ax = axes[0, 1]
    for name, margin in margins.items():
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(margin)
            x_plot = np.linspace(margin.min(), margin.max(), 200)
            ax.plot(x_plot, kde(x_plot), label=name, linewidth=2)
        except:
            pass
    ax.set_xlabel('Margin', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Margin KDE', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # AUROC for margin as separator
    ax = axes[1, 0]
    aurocs = {}
    for name in margins.keys():
        margin_vals = margins[name]
        correct_vals = is_correct[name].astype(float)
        try:
            auroc = roc_auc_score(correct_vals, margin_vals)
            aurocs[name] = auroc
        except:
            aurocs[name] = 0.0
    
    names_list = list(aurocs.keys())
    auroc_vals = [aurocs[n] for n in names_list]
    colors_bar = ['steelblue'] * E + ['orange', 'green']
    ax.bar(range(len(names_list)), auroc_vals, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(names_list)))
    ax.set_xticklabels(names_list, rotation=15, ha='right')
    ax.set_ylabel('AUROC (Margin vs Correctness)', fontsize=11)
    ax.set_title('Margin Discriminative Power', fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Random (0.5)')
    ax.legend()
    
    for i, auroc in enumerate(auroc_vals):
        ax.text(i, auroc + 0.01, f'{auroc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Jaccard stability
    ax = axes[1, 1]
    theta_values = np.linspace(0.0, 0.5, 20)
    jaccard_scores = {}
    
    for name in ['Uniform-Mix', 'Gating-Mix']:
        margin_vals = margins[name]
        jaccards = []
        
        for i in range(len(theta_values) - 1):
            theta1, theta2 = theta_values[i], theta_values[i+1]
            accept1 = (margin_vals >= theta1).astype(int)
            accept2 = (margin_vals >= theta2).astype(int)
            
            # Jaccard = intersection / union
            intersection = (accept1 & accept2).sum()
            union = (accept1 | accept2).sum()
            if union > 0:
                jaccard = intersection / union
            else:
                jaccard = 1.0
            
            jaccards.append(jaccard)
        
        jaccard_scores[name] = jaccards
        ax.plot(theta_values[1:], jaccards, 'o-', label=name, linewidth=2, markersize=4)
    
    ax.set_xlabel('Threshold θ', fontsize=11)
    ax.set_ylabel('Jaccard Index', fontsize=11)
    ax.set_title('Acceptance Set Stability', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle('Margin & Ranking Stability Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return aurocs, jaccard_scores


# ================================
# C. Disagreement & Entropy
# ================================
def compute_disagreement(expert_posteriors: torch.Tensor) -> torch.Tensor:
    """Compute disagreement = 1 - max over classes of mean one-hot votes."""
    # Mean posterior across experts
    mean_posterior = expert_posteriors.mean(dim=1)  # [N, C]
    # Disagreement = 1 - max(mean posterior)
    disagreement = 1.0 - mean_posterior.max(dim=-1)[0]  # [N]
    return disagreement


def plot_disagreement_analysis(expert_posteriors: torch.Tensor, uniform_mixture: torch.Tensor,
                               gating_mixture: torch.Tensor, labels: torch.Tensor,
                               class_to_group: torch.Tensor, save_path: Path):
    """Plot disagreement vs error/uncertainty."""
    N, E, C = expert_posteriors.shape
    
    # Compute disagreement
    disagreement = compute_disagreement(expert_posteriors).cpu().numpy()
    
    # Compute errors (correctness)
    uniform_preds = uniform_mixture.argmax(dim=-1)
    is_correct = (uniform_preds == labels).cpu().numpy()
    errors = 1.0 - is_correct.astype(float)
    
    # Compute entropy
    eps = 1e-8
    uniform_entropy = -torch.sum(uniform_mixture * torch.log(uniform_mixture + eps), dim=-1).cpu().numpy()
    mean_expert_entropy = -torch.sum(
        expert_posteriors * torch.log(expert_posteriors + eps), dim=-1
    ).mean(dim=1).cpu().numpy()
    
    # Group labels
    groups = class_to_group[labels].cpu().numpy()
    head_mask = (groups == 0)
    tail_mask = (groups == 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Scatter: Disagreement vs Error (head/tail)
    ax = axes[0, 0]
    ax.scatter(disagreement[head_mask], errors[head_mask], alpha=0.3, s=10, label='Head', color='skyblue')
    ax.scatter(disagreement[tail_mask], errors[tail_mask], alpha=0.3, s=10, label='Tail', color='coral')
    ax.set_xlabel('Disagreement', fontsize=11)
    ax.set_ylabel('Error', fontsize=11)
    ax.set_title('Disagreement vs Error (by Group)', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Scatter: Disagreement vs Entropy
    ax = axes[0, 1]
    ax.scatter(disagreement[head_mask], uniform_entropy[head_mask], alpha=0.3, s=10, label='Head', color='skyblue')
    ax.scatter(disagreement[tail_mask], uniform_entropy[tail_mask], alpha=0.3, s=10, label='Tail', color='coral')
    ax.set_xlabel('Disagreement', fontsize=11)
    ax.set_ylabel('Mixture Entropy', fontsize=11)
    ax.set_title('Disagreement vs Entropy', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Histogram: Mixture Entropy vs Mean Expert Entropy
    ax = axes[1, 0]
    ax.hist(mean_expert_entropy[head_mask], bins=50, alpha=0.5, label='Mean Expert Entropy (Head)', 
           color='lightblue', density=True)
    ax.hist(uniform_entropy[head_mask], bins=50, alpha=0.5, label='Mixture Entropy (Head)', 
           color='skyblue', density=True)
    ax.hist(mean_expert_entropy[tail_mask], bins=50, alpha=0.5, label='Mean Expert Entropy (Tail)', 
           color='lightsalmon', density=True)
    ax.hist(uniform_entropy[tail_mask], bins=50, alpha=0.5, label='Mixture Entropy (Tail)', 
           color='coral', density=True)
    ax.set_xlabel('Entropy', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Entropy: Mixture vs Experts (by Group)', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Correlation summary
    ax = axes[1, 1]
    ax.axis('off')
    
    corr_disagree_error = np.corrcoef(disagreement, errors)[0, 1]
    corr_disagree_entropy = np.corrcoef(disagreement, uniform_entropy)[0, 1]
    
    stats_text = f"""
    Disagreement Analysis:
    ----------------------
    Correlation (Disagreement, Error): {corr_disagree_error:.4f}
    Correlation (Disagreement, Entropy): {corr_disagree_entropy:.4f}
    
    Mean Disagreement:
      Head: {disagreement[head_mask].mean():.4f}
      Tail: {disagreement[tail_mask].mean():.4f}
    
    Mean Mixture Entropy:
      Head: {uniform_entropy[head_mask].mean():.4f}
      Tail: {uniform_entropy[tail_mask].mean():.4f}
    
    Mean Expert Entropy:
      Head: {mean_expert_entropy[head_mask].mean():.4f}
      Tail: {mean_expert_entropy[tail_mask].mean():.4f}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
           verticalalignment='center')
    
    plt.suptitle('Disagreement & Entropy Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ================================
# D. Per-Group RC Curves
# ================================
def load_rc_results() -> Dict:
    """Load RC curve results."""
    results_path = Path(CFG.results_dir) / "ltr_plugin_gating_balanced.json"
    if not results_path.exists():
        return {}
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    return results


def plot_per_group_rc(rc_results: Dict, save_path: Path):
    """Plot per-group RC curves: error_k(ρ) and AURC_k."""
    if 'rc_curve' not in rc_results or 'test' not in rc_results['rc_curve']:
        print("Warning: RC curve data not found")
        return
    
    test_rc = rc_results['rc_curve']['test']
    rejection_rates = np.array(test_rc['rejection_rates'])
    group_errors_list = test_rc.get('group_errors_list', [])
    
    if len(group_errors_list) == 0:
        print("Warning: Group errors not found in RC results")
        return
    
    head_errors = np.array([ge[0] if len(ge) > 0 else 1.0 for ge in group_errors_list])
    tail_errors = np.array([ge[1] if len(ge) > 1 else 1.0 for ge in group_errors_list])
    balanced_errors = np.array(test_rc['balanced_errors'])
    
    # Sort by rejection rate
    sort_idx = np.argsort(rejection_rates)
    rejection_rates = rejection_rates[sort_idx]
    head_errors = head_errors[sort_idx]
    tail_errors = tail_errors[sort_idx]
    balanced_errors = balanced_errors[sort_idx]
    
    # Compute AURC per group
    aurc_head = np.trapz(head_errors, rejection_rates)
    aurc_tail = np.trapz(tail_errors, rejection_rates)
    aurc_balanced = np.trapz(balanced_errors, rejection_rates)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Per-group RC curves
    axes[0].plot(rejection_rates, head_errors, 'o-', color='skyblue', 
                label=f'Head (AURC={aurc_head:.4f})', linewidth=2)
    axes[0].plot(rejection_rates, tail_errors, 's-', color='coral',
                label=f'Tail (AURC={aurc_tail:.4f})', linewidth=2)
    axes[0].plot(rejection_rates, balanced_errors, 'd-', color='green',
                label=f'Balanced (AURC={aurc_balanced:.4f})', linewidth=2)
    axes[0].set_xlabel('Proportion of Rejections ρ', fontsize=12)
    axes[0].set_ylabel('Group Error', fontsize=12)
    axes[0].set_title('Per-Group RC Curves', fontweight='bold', fontsize=13)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([0, 1])
    
    # AURC comparison
    methods = ['Head', 'Tail', 'Balanced']
    aurcs = [aurc_head, aurc_tail, aurc_balanced]
    colors_bar = ['skyblue', 'coral', 'green']
    
    axes[1].bar(methods, aurcs, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('AURC', fontsize=12)
    axes[1].set_title('AURC by Group', fontweight='bold', fontsize=13)
    axes[1].grid(alpha=0.3, axis='y')
    
    for i, (method, aurc) in enumerate(zip(methods, aurcs)):
        axes[1].text(i, aurc + 0.005, f'{aurc:.4f}', ha='center', va='bottom', fontsize=11)
    
    plt.suptitle('Per-Group RC Curves & AURC', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ================================
# E. Fairness by Coverage (α-coverage per group)
# ================================
def plot_alpha_coverage(rc_results: Dict, save_path: Path):
    """Plot α̂_k vs ρ and difference α̂_tail - α̂_head."""
    if 'results_per_cost' not in rc_results:
        print("Warning: results_per_cost not found")
        return
    
    results_per_cost = rc_results['results_per_cost']
    
    # Extract alpha values and rejection rates
    rejection_rates = []
    alpha_head_list = []
    alpha_tail_list = []
    
    for result in results_per_cost:
        rejection_rates.append(1.0 - result['val_metrics']['coverage'])
        alpha = result['alpha']
        if len(alpha) >= 2:
            alpha_head_list.append(alpha[0])
            alpha_tail_list.append(alpha[1])
        else:
            alpha_head_list.append(None)
            alpha_tail_list.append(None)
    
    rejection_rates = np.array(rejection_rates)
    alpha_head = np.array([a for a in alpha_head_list if a is not None])
    alpha_tail = np.array([a for a in alpha_tail_list if a is not None])
    
    if len(alpha_head) == 0 or len(alpha_tail) == 0:
        print("Warning: Alpha values not found")
        return
    
    # Sort by rejection rate
    sort_idx = np.argsort(rejection_rates[:len(alpha_head)])
    rejection_rates_sorted = rejection_rates[:len(alpha_head)][sort_idx]
    alpha_head_sorted = alpha_head[sort_idx]
    alpha_tail_sorted = alpha_tail[sort_idx]
    
    gap = alpha_tail_sorted - alpha_head_sorted
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # α̂_k vs ρ
    axes[0].plot(rejection_rates_sorted, alpha_head_sorted, 'o-', color='skyblue',
                label='α̂_head', linewidth=2, markersize=6)
    axes[0].plot(rejection_rates_sorted, alpha_tail_sorted, 's-', color='coral',
                label='α̂_tail', linewidth=2, markersize=6)
    axes[0].set_xlabel('Proportion of Rejections ρ', fontsize=12)
    axes[0].set_ylabel('α̂_k (Coverage per Group)', fontsize=12)
    axes[0].set_title('Group Coverage α̂_k vs Rejection Rate', fontweight='bold', fontsize=13)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([0, 1])
    
    # Gap: α̂_tail - α̂_head
    axes[1].plot(rejection_rates_sorted, gap, 'd-', color='purple', linewidth=2, markersize=6)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Proportion of Rejections ρ', fontsize=12)
    axes[1].set_ylabel('α̂_tail - α̂_head', fontsize=12)
    axes[1].set_title('Coverage Gap (Tail - Head)', fontweight='bold', fontsize=13)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim([0, 1])
    
    plt.suptitle('Fairness by Coverage: α-coverage per Group', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ================================
# F. Who Gets Rejected (Rejection Heatmap)
# ================================
def plot_rejection_heatmap(mixture_posterior: torch.Tensor, labels: torch.Tensor,
                          class_to_group: torch.Tensor, save_path: Path):
    """Plot rejection heatmap per class."""
    from src.models.ltr_plugin import LtRPlugin, LtRPluginConfig
    
    # Get device from input tensors
    device = mixture_posterior.device
    
    # Load RC results to get optimal parameters
    rc_results = load_rc_results()
    if 'results_per_cost' not in rc_results or len(rc_results['results_per_cost']) == 0:
        print("Warning: Cannot create rejection heatmap without RC results")
        return
    
    # Use parameters from first RC point
    first_result = rc_results['results_per_cost'][0]
    alpha = np.array(first_result['alpha'])
    mu = np.array(first_result['mu'])
    cost = first_result.get('cost_test', 0.0)
    
    # Create plugin
    config = LtRPluginConfig(
        num_classes=CFG.num_classes,
        num_groups=CFG.num_groups,
        group_boundaries=[69],  # Head/tail split
        param_mode='group'
    )
    plugin = LtRPlugin(config).to(device)  # Move plugin to same device
    
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)  # Create on correct device
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)  # Create on correct device
    plugin.set_parameters(alpha=alpha_t, mu=mu_t, cost=cost)
    
    # Ensure class_to_group is on same device
    class_to_group = class_to_group.to(device)
    
    # Compute rejections at different thresholds
    with torch.no_grad():
        reject = plugin.predict_reject(mixture_posterior)
    
    # Compute rejection rate per class
    num_classes = CFG.num_classes
    rejection_by_class = np.zeros(num_classes)
    count_by_class = np.zeros(num_classes)
    
    labels_np = labels.cpu().numpy()
    reject_np = reject.cpu().numpy()
    
    for c in range(num_classes):
        mask = (labels_np == c)
        count_by_class[c] = mask.sum()
        if count_by_class[c] > 0:
            rejection_by_class[c] = reject_np[mask].mean()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Reshape for heatmap (10x10 grid for 100 classes)
    heatmap_data = rejection_by_class.reshape(10, 10)
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Class Index (mod 10)', fontsize=12)
    ax.set_ylabel('Class Index (div 10)', fontsize=12)
    ax.set_title('Rejection Rate by Class (Heatmap)', fontweight='bold', fontsize=13)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rejection Rate', fontsize=11)
    
    # Add head/tail boundary line
    head_tail_boundary = 69 // 10  # 6.9 -> row 6
    ax.axhline(head_tail_boundary - 0.5, color='blue', linestyle='--', linewidth=2, 
              label='Head/Tail Boundary')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ================================
# G. Gating Weight Distribution by Group
# ================================
def plot_gating_weight_distribution(gating_weights: torch.Tensor, labels: torch.Tensor,
                                   class_to_group: torch.Tensor, expert_names: List[str],
                                   expert_posteriors: torch.Tensor, save_path: Path):
    """Plot gating weight distribution by group and entropy/margin bins."""
    N, E = gating_weights.shape
    groups = class_to_group[labels]
    head_mask = (groups == 0)
    tail_mask = (groups == 1)
    
    # Compute entropy and margin for binning
    uniform_mixture = expert_posteriors.mean(dim=1)
    eps = 1e-8
    entropy = -torch.sum(uniform_mixture * torch.log(uniform_mixture + eps), dim=-1).cpu().numpy()
    margin = compute_margin(uniform_mixture).cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Boxplot: w_e(x) by head/tail
    ax = axes[0, 0]
    data_to_plot = []
    labels_box = []
    
    for e in range(E):
        weights_head = gating_weights[head_mask, e].cpu().numpy()
        weights_tail = gating_weights[tail_mask, e].cpu().numpy()
        data_to_plot.extend([weights_head, weights_tail])
        labels_box.extend([f'{expert_names[e]}\nHead', f'{expert_names[e]}\nTail'])
    
    bp = ax.boxplot(data_to_plot, labels=labels_box, patch_artist=True)
    colors_box = ['lightblue', 'lightcoral'] * E
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Gating Weight w_e(x)', fontsize=11)
    ax.set_title('Gating Weights by Group', fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(alpha=0.3, axis='y')
    
    # Correlation: w_e vs group indicator
    ax = axes[0, 1]
    correlations = {}
    for e in range(E):
        weights_e = gating_weights[:, e].cpu().numpy()
        is_tail = (groups == 1).cpu().numpy().astype(float)
        corr_tail = np.corrcoef(weights_e, is_tail)[0, 1]
        
        is_head = (groups == 0).cpu().numpy().astype(float)
        corr_head = np.corrcoef(weights_e, is_head)[0, 1]
        
        correlations[expert_names[e]] = {'head': corr_head, 'tail': corr_tail}
    
    x = np.arange(E)
    corr_heads = [correlations[n]['head'] for n in expert_names]
    corr_tails = [correlations[n]['tail'] for n in expert_names]
    
    width = 0.35
    ax.bar(x - width/2, corr_heads, width, label='Corr(w_e, Head)', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, corr_tails, width, label='Corr(w_e, Tail)', alpha=0.8, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(expert_names, rotation=15, ha='right')
    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_title('Gating Weight vs Group Correlation', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Weights by entropy bins
    ax = axes[1, 0]
    entropy_bins = np.linspace(entropy.min(), entropy.max(), 5)
    bin_labels = [f'{entropy_bins[i]:.2f}-{entropy_bins[i+1]:.2f}' for i in range(len(entropy_bins)-1)]
    
    mean_weights_by_entropy = {name: [] for name in expert_names}
    for i in range(len(entropy_bins) - 1):
        mask = (entropy >= entropy_bins[i]) & (entropy < entropy_bins[i+1])
        if mask.sum() > 0:
            for e, name in enumerate(expert_names):
                mean_w = gating_weights[mask, e].mean().item()
                mean_weights_by_entropy[name].append(mean_w)
        else:
            for name in expert_names:
                mean_weights_by_entropy[name].append(0.0)
    
    x = np.arange(len(bin_labels))
    width = 0.25
    for i, name in enumerate(expert_names):
        offset = (i - 1) * width
        ax.bar(x + offset, mean_weights_by_entropy[name], width, label=name, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=15, ha='right')
    ax.set_ylabel('Mean Gating Weight', fontsize=11)
    ax.set_xlabel('Entropy Bin', fontsize=11)
    ax.set_title('Gating Weights by Entropy Bins', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Weights by margin bins
    ax = axes[1, 1]
    margin_bins = np.linspace(margin.min(), margin.max(), 5)
    bin_labels_margin = [f'{margin_bins[i]:.2f}-{margin_bins[i+1]:.2f}' for i in range(len(margin_bins)-1)]
    
    mean_weights_by_margin = {name: [] for name in expert_names}
    for i in range(len(margin_bins) - 1):
        mask = (margin >= margin_bins[i]) & (margin < margin_bins[i+1])
        if mask.sum() > 0:
            for e, name in enumerate(expert_names):
                mean_w = gating_weights[mask, e].mean().item()
                mean_weights_by_margin[name].append(mean_w)
        else:
            for name in expert_names:
                mean_weights_by_margin[name].append(0.0)
    
    x = np.arange(len(bin_labels_margin))
    for i, name in enumerate(expert_names):
        offset = (i - 1) * width
        ax.bar(x + offset, mean_weights_by_margin[name], width, label=name, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels_margin, rotation=15, ha='right')
    ax.set_ylabel('Mean Gating Weight', fontsize=11)
    ax.set_xlabel('Margin Bin', fontsize=11)
    ax.set_title('Gating Weights by Margin Bins', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Gating Weight Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return correlations


# ================================
# Main Function
# ================================
def main():
    """Main analysis function."""
    print("="*70)
    print("Detailed MoE-L2R Analysis - Extended Plots")
    print("="*70)
    
    # Setup
    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    expert_logits_test = load_expert_logits(CFG.expert_names, "test", CFG.device)
    labels_test = load_labels("test", CFG.device)
    expert_posteriors_test = F.softmax(expert_logits_test, dim=-1)
    
    print(f"   Loaded {expert_logits_test.shape[0]} test samples")
    
    # Load gating
    print("\n2. Loading gating network...")
    gating = load_gating_network(CFG.device)
    
    # Compute mixture
    print("\n3. Computing mixture posteriors...")
    mixture_posterior, gating_weights = compute_mixture_posterior(
        expert_logits_test, gating, CFG.device
    )
    uniform_mixture = expert_posteriors_test.mean(dim=1)
    
    # Build class-to-group mapping
    print("\n4. Building class-to-group mapping...")
    class_to_group = build_class_to_group(CFG.device)
    
    # Load RC results
    print("\n5. Loading RC results...")
    rc_results = load_rc_results()
    
    # ================================
    # A. Reliability & NLL Decomposition
    # ================================
    print("\n" + "="*70)
    print("(A) Reliability & NLL Decomposition")
    print("="*70)
    
    print("   Plotting NLL decomposition...")
    plot_nll_decomposition(
        expert_posteriors_test, uniform_mixture, mixture_posterior,
        labels_test, CFG.expert_names,
        output_dir / "nll_decomposition.png"
    )
    
    print("   Plotting calibration vs coverage...")
    for method_name, posteriors in [
        ('Uniform-Mix', uniform_mixture),
        ('Gating-Mix', mixture_posterior)
    ]:
        plot_calibration_vs_coverage(
            posteriors, labels_test, method_name,
            output_dir / f"calibration_vs_coverage_{method_name.lower().replace('-', '_')}.png"
        )
    
    # ================================
    # B. Margin & Ranking Stability
    # ================================
    print("\n" + "="*70)
    print("(B) Margin & Ranking Stability")
    print("="*70)
    
    print("   Plotting margin analysis...")
    aurocs, jaccard_scores = plot_margin_analysis(
        expert_posteriors_test, uniform_mixture, mixture_posterior,
        labels_test, CFG.expert_names,
        output_dir / "margin_analysis.png"
    )
    
    # ================================
    # C. Disagreement & Entropy
    # ================================
    print("\n" + "="*70)
    print("(C) Disagreement & Entropy")
    print("="*70)
    
    print("   Plotting disagreement analysis...")
    plot_disagreement_analysis(
        expert_posteriors_test, uniform_mixture, mixture_posterior,
        labels_test, class_to_group,
        output_dir / "disagreement_analysis.png"
    )
    
    # ================================
    # D. Per-Group RC Curves
    # ================================
    print("\n" + "="*70)
    print("(D) Per-Group RC Curves")
    print("="*70)
    
    if rc_results:
        print("   Plotting per-group RC curves...")
        plot_per_group_rc(rc_results, output_dir / "per_group_rc.png")
    else:
        print("   Warning: RC results not available")
    
    # ================================
    # E. Fairness by Coverage
    # ================================
    print("\n" + "="*70)
    print("(E) Fairness by Coverage")
    print("="*70)
    
    if rc_results:
        print("   Plotting alpha coverage...")
        plot_alpha_coverage(rc_results, output_dir / "alpha_coverage.png")
    else:
        print("   Warning: RC results not available")
    
    # ================================
    # F. Rejection Heatmap
    # ================================
    print("\n" + "="*70)
    print("(F) Rejection Heatmap")
    print("="*70)
    
    print("   Plotting rejection heatmap...")
    plot_rejection_heatmap(
        mixture_posterior, labels_test, class_to_group,
        output_dir / "rejection_heatmap.png"
    )
    
    # ================================
    # G. Gating Weight Distribution
    # ================================
    print("\n" + "="*70)
    print("(G) Gating Weight Distribution")
    print("="*70)
    
    print("   Plotting gating weight distribution...")
    correlations = plot_gating_weight_distribution(
        gating_weights, labels_test, class_to_group,
        CFG.expert_names, expert_posteriors_test,
        output_dir / "gating_weight_distribution.png"
    )
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

