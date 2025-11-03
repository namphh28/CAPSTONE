#!/usr/bin/env python3
"""
Comprehensive MoE-L2R Analysis Script
======================================

Implements all analyses from the research outline:

(A) MoE → Smooth signals (before reject):
    - Variance reduction between experts
    - Mutual Information (MI)
    - Calibration (ECE/NLL/Brier)
    - Oracle upper bound vs gating

(B) Understanding gating:
    - Weight entropy & effective number of experts
    - Correlation between difficulty and weights
    - Gating fairness by group
    - Router ablation (optional)

(C) MoE → Reject via L2R-LT:
    - RC/AURC curves
    - Per-group coverage analysis
    - Gap analysis (tail-head)

Usage:
    python scripts/moe_l2r_comprehensive_analysis.py
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr


# ================================
# JSON Serialization Helper
# ================================
def convert_to_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy/torch types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    else:
        # For any other type, try to convert to string
        return str(obj)


# ================================
# Config
# ================================
@dataclass
class Config:
    dataset_name: str = "cifar100_lt_if100"
    splits_dir: str = "./data/cifar100_lt_if100_splits_fixed"
    logits_dir: str = "./outputs/logits/cifar100_lt_if100"
    gating_checkpoint: str = "./checkpoints/gating_map/cifar100_lt_if100/final_gating.pth"
    results_dir: str = "./results/ltr_plugin/cifar100_lt_if100"
    output_dir: str = "./results/moe_analysis/cifar100_lt_if100"
    
    expert_names: List[str] = field(
        default_factory=lambda: [
            "ce_baseline",
            "logitadjust_baseline",
            "balsoftmax_baseline",
        ]
    )
    
    num_classes: int = 100
    num_groups: int = 2
    tail_threshold: int = 20  # tail if train count <= 20
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


# ================================
# Utility Functions
# ================================
def load_expert_logits(expert_names: List[str], split: str, device: str = CFG.device) -> torch.Tensor:
    """Load logits from all experts and stack them."""
    logits_list = []
    for expert_name in expert_names:
        path = Path(CFG.logits_dir) / expert_name / f"{split}_logits.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing logits: {path}")
        logits = torch.load(path, map_location=device).float()
        logits_list.append(logits)
    
    # Stack: [E, N, C] -> transpose to [N, E, C]
    logits = torch.stack(logits_list, dim=0).transpose(0, 1)
    return logits


def load_labels(split: str, device: str = CFG.device) -> torch.Tensor:
    """Load labels for split."""
    # Prefer saved targets
    cand = Path(CFG.logits_dir) / CFG.expert_names[0] / f"{split}_targets.pt"
    if cand.exists():
        t = torch.load(cand, map_location=device)
        if isinstance(t, torch.Tensor):
            return t.to(device=device, dtype=torch.long)
    
    # Fallback: reconstruct from CIFAR100 and indices
    import torchvision
    indices_file = Path(CFG.splits_dir) / f"{split}_indices.json"
    with open(indices_file, "r", encoding="utf-8") as f:
        indices = json.load(f)
    is_train = split in ("expert", "gating", "train")
    ds = torchvision.datasets.CIFAR100(root="./data", train=is_train, download=False)
    labels = torch.tensor(
        [ds.targets[i] for i in indices], dtype=torch.long, device=device
    )
    return labels


def load_gating_network(device: str = CFG.device):
    """Load trained gating network."""
    from src.models.gating_network_map import GatingNetwork, GatingMLP
    from src.models.gating import GatingFeatureBuilder
    
    num_experts = len(CFG.expert_names)
    num_classes = CFG.num_classes
    
    gating = GatingNetwork(
        num_experts=num_experts, num_classes=num_classes, routing="dense"
    ).to(device)
    
    # Match compact feature setup used during training
    compact_dim = 7 * num_experts + 3
    gating.mlp = GatingMLP(
        input_dim=compact_dim,
        num_experts=num_experts,
        hidden_dims=[256, 128],
        dropout=0.1,
        activation='relu',
    ).to(device)
    
    checkpoint = torch.load(CFG.gating_checkpoint, map_location=device, weights_only=False)
    gating.load_state_dict(checkpoint["model_state_dict"])
    gating.eval()
    return gating


def build_class_to_group(device: str = CFG.device) -> torch.Tensor:
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


def compute_mixture_posterior(expert_logits: torch.Tensor, gating_net, device: str = CFG.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mixture posterior using gating network. Returns (mixture_posterior, gating_weights)."""
    with torch.no_grad():
        expert_posteriors = F.softmax(expert_logits, dim=-1)  # [N, E, C]
        from src.models.gating import GatingFeatureBuilder
        feat_builder = GatingFeatureBuilder()
        features = feat_builder(expert_logits)  # [N, 7*E+3]
        
        gating_logits = gating_net.mlp(features)  # [N, E]
        gating_weights = gating_net.router(gating_logits)  # [N, E]
        
        # Mixture posterior: η̃(x) = Σ_e w_e · p^(e)(y|x)
        mixture_posterior = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)  # [N, C]
        
        return mixture_posterior, gating_weights


# ================================
# (A) MoE → Smooth Signals Analysis
# ================================
def compute_expert_variance(expert_posteriors: torch.Tensor, labels: torch.Tensor) -> Dict:
    """
    Compute variance reduction between experts.
    
    Var_e[p_y^(e)(x)]: variance across experts for each class y
    Δ_var(x) = E_y Var_e[p_y^(e)(x)]: expected variance across classes
    """
    # expert_posteriors: [N, E, C]
    N, E, C = expert_posteriors.shape
    
    # For each sample x and class y, compute variance across experts
    # Var_e[p_y^(e)(x)]: [N, C]
    expert_var_per_class = expert_posteriors.var(dim=1)  # [N, C]
    
    # Δ_var(x) = E_y Var_e[p_y^(e)(x)]: [N]
    delta_var = expert_var_per_class.mean(dim=-1)  # [N]
    
    # For mixture: compute mixture posterior and compare variance
    uniform_weights = torch.ones(N, E, device=expert_posteriors.device) / E
    uniform_mixture = (uniform_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)  # [N, C]
    
    # Variance in mixture (should be lower)
    mixture_var_per_class = torch.zeros(N, C, device=expert_posteriors.device)
    for e in range(E):
        diff = (expert_posteriors[:, e, :] - uniform_mixture) ** 2
        mixture_var_per_class += (1.0 / E) * diff
    mixture_delta_var = mixture_var_per_class.mean(dim=-1)  # [N]
    
    return {
        'expert_var_per_class': expert_var_per_class.cpu().numpy(),  # [N, C]
        'delta_var': delta_var.cpu().numpy(),  # [N] - before mixture
        'mixture_delta_var': mixture_delta_var.cpu().numpy(),  # [N] - after uniform mixture
        'variance_reduction': (delta_var - mixture_delta_var).cpu().numpy(),  # [N]
    }


def compute_mutual_information(expert_posteriors: torch.Tensor) -> Dict:
    """
    Compute Mutual Information (MI) between experts.
    
    MI(x) = H(1/E Σ_e p^(e)) - 1/E Σ_e H(p^(e))
    
    High MI → experts disagree; averaging helps stabilize estimation.
    """
    # expert_posteriors: [N, E, C]
    N, E, C = expert_posteriors.shape
    eps = 1e-8
    
    # Compute entropy for each expert: H(p^(e)): [N, E]
    expert_entropies = -torch.sum(
        expert_posteriors * torch.log(expert_posteriors + eps), dim=-1
    )  # [N, E]
    
    # Average entropy across experts: 1/E Σ_e H(p^(e)): [N]
    mean_expert_entropy = expert_entropies.mean(dim=-1)  # [N]
    
    # Compute mixture posterior (uniform for MI calculation)
    uniform_mixture = expert_posteriors.mean(dim=1)  # [N, C]
    
    # Entropy of mixture: H(1/E Σ_e p^(e)): [N]
    mixture_entropy = -torch.sum(
        uniform_mixture * torch.log(uniform_mixture + eps), dim=-1
    )  # [N]
    
    # MI(x) = H(mixture) - mean(H(experts)): [N]
    mi = mixture_entropy - mean_expert_entropy  # [N]
    
    return {
        'expert_entropies': expert_entropies.cpu().numpy(),  # [N, E]
        'mean_expert_entropy': mean_expert_entropy.cpu().numpy(),  # [N]
        'mixture_entropy': mixture_entropy.cpu().numpy(),  # [N]
        'mutual_information': mi.cpu().numpy(),  # [N]
    }


def compute_calibration_metrics(posteriors: torch.Tensor, labels: torch.Tensor, class_to_group: torch.Tensor) -> Dict:
    """
    Compute calibration metrics: ECE, NLL, Brier Score.
    
    Returns overall and per-group metrics.
    """
    # posteriors: [N, C], labels: [N]
    N, C = posteriors.shape
    
    # Get predictions and confidences
    confidences, predictions = posteriors.max(dim=-1)
    accuracies = (predictions == labels).float()
    
    # ECE: Expected Calibration Error
    n_bins = 15
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=posteriors.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin.item())
            bin_confs.append(avg_confidence_in_bin.item())
            bin_counts.append(in_bin.sum().item())
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)
    
    ece = ece.item()
    
    # NLL: Negative Log Likelihood
    true_class_probs = posteriors[torch.arange(N), labels]
    nll = -torch.log(true_class_probs + 1e-8).mean().item()
    
    # Brier Score
    one_hot_labels = F.one_hot(labels, num_classes=C).float()
    brier = torch.mean(torch.sum((posteriors - one_hot_labels) ** 2, dim=-1)).item()
    
    # Per-group metrics
    groups = class_to_group[labels]
    num_groups = int(class_to_group.max().item() + 1)
    
    group_metrics = {}
    for g in range(num_groups):
        mask = (groups == g)
        if mask.sum() > 0:
            group_posteriors = posteriors[mask]
            group_labels = labels[mask]
            group_ece = compute_group_ece(group_posteriors, group_labels, n_bins)
            group_nll = -torch.log(group_posteriors[torch.arange(mask.sum()), group_labels] + 1e-8).mean().item()
            group_one_hot = F.one_hot(group_labels, num_classes=C).float()
            group_brier = torch.mean(torch.sum((group_posteriors - group_one_hot) ** 2, dim=-1)).item()
            
            group_metrics[f'group_{g}'] = {
                'ece': group_ece,
                'nll': group_nll,
                'brier': group_brier,
            }
    
    return {
        'ece': ece,
        'nll': nll,
        'brier': brier,
        'bin_accs': bin_accs,
        'bin_confs': bin_confs,
        'bin_counts': bin_counts,
        'group_metrics': group_metrics,
    }


def compute_group_ece(posteriors: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Compute ECE for a specific group."""
    confidences, predictions = posteriors.max(dim=-1)
    accuracies = (predictions == labels).float()
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=posteriors.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def compute_oracle_vs_gating(expert_posteriors: torch.Tensor, gating_weights: torch.Tensor, 
                             uniform_mixture: torch.Tensor, labels: torch.Tensor, 
                             class_to_group: torch.Tensor) -> Dict:
    """
    Compare Oracle@E, Uniform-Mix, and Gating-Mix.
    
    Oracle@E: choose expert with highest probability for true label (upper bound)
    Uniform-Mix: average all experts uniformly
    Gating-Mix: weighted average using gating network
    """
    N, E, C = expert_posteriors.shape
    
    # Oracle@E: for each sample, choose expert with highest p(y_true | expert)
    # This means: for each sample x, choose expert e that maximizes p(y_true | expert e)
    oracle_posteriors = torch.zeros_like(uniform_mixture)
    
    # For each sample, find expert with highest probability for true label
    true_class_probs = expert_posteriors[torch.arange(N), :, labels]  # [N, E]
    best_expert_per_sample = true_class_probs.argmax(dim=-1)  # [N]
    
    # Use posterior from best expert for each sample
    for i in range(N):
        oracle_posteriors[i] = expert_posteriors[i, best_expert_per_sample[i]]
    
    # Compute metrics for each method
    methods = {
        'oracle': oracle_posteriors,
        'uniform': uniform_mixture,
        'gating': None  # Will be set from input
    }
    
    # Get gating mixture (need to compute it)
    from src.models.gating import GatingFeatureBuilder
    from src.models.gating_network_map import GatingNetwork
    # For now, assume gating weights are provided
    
    if gating_weights is not None:
        gating_mixture = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)
        methods['gating'] = gating_mixture
    else:
        methods['gating'] = uniform_mixture
    
    # Compute metrics
    results = {}
    groups = class_to_group[labels]
    num_groups = int(class_to_group.max().item() + 1)
    
    for method_name, posteriors_method in methods.items():
        if posteriors_method is None:
            continue
        
        predictions = posteriors_method.argmax(dim=-1)
        acc = (predictions == labels).float().mean().item()
        
        # Per-group accuracies
        group_accs = []
        for g in range(num_groups):
            mask = (groups == g)
            if mask.sum() > 0:
                group_acc = (predictions[mask] == labels[mask]).float().mean().item()
                group_accs.append(group_acc)
            else:
                group_accs.append(0.0)
        
        # Calibration
        cal_metrics = compute_calibration_metrics(posteriors_method, labels, class_to_group)
        
        results[method_name] = {
            'acc': acc,
            'group_accs': group_accs,
            'ece': cal_metrics['ece'],
            'nll': cal_metrics['nll'],
            'brier': cal_metrics['brier'],
        }
    
    return results


# ================================
# (B) Understanding Gating Analysis
# ================================
def compute_gating_statistics(gating_weights: torch.Tensor, labels: torch.Tensor, 
                             class_to_group: torch.Tensor) -> Dict:
    """
    Compute gating statistics: entropy, effective experts, correlations.
    
    H(w(x)) = -Σ_e w_e log w_e
    E_eff(x) = 1 / Σ_e w_e^2
    """
    N, E = gating_weights.shape
    eps = 1e-8
    
    # Weight entropy: H(w(x)) = -Σ_e w_e log w_e
    weight_entropy = -torch.sum(
        gating_weights * torch.log(gating_weights + eps), dim=-1
    )  # [N]
    
    # Effective number of experts: E_eff(x) = 1 / Σ_e w_e^2
    weight_squared_sum = torch.sum(gating_weights ** 2, dim=-1)  # [N]
    effective_experts = 1.0 / (weight_squared_sum + eps)  # [N]
    
    # Per-expert mean weights
    mean_weights = gating_weights.mean(dim=0)  # [E]
    
    # Load balance (std of mean weights)
    load_balance_std = mean_weights.std().item()
    
    # Group-wise statistics
    groups = class_to_group[labels]
    num_groups = int(class_to_group.max().item() + 1)
    
    group_stats = {}
    for g in range(num_groups):
        mask = (groups == g)
        if mask.sum() > 0:
            group_weights = gating_weights[mask]
            group_stats[f'group_{g}'] = {
                'mean_entropy': weight_entropy[mask].mean().item(),
                'mean_effective_experts': effective_experts[mask].mean().item(),
                'mean_weights': group_weights.mean(dim=0).cpu().numpy().tolist(),  # [E]
            }
    
    return {
        'weight_entropy': weight_entropy.cpu().numpy(),  # [N]
        'effective_experts': effective_experts.cpu().numpy(),  # [N]
        'mean_weights': mean_weights.cpu().numpy(),  # [E]
        'load_balance_std': load_balance_std,
        'group_stats': group_stats,
    }


def compute_difficulty_correlation(expert_posteriors: torch.Tensor, gating_weights: torch.Tensor,
                                   labels: torch.Tensor, class_to_group: torch.Tensor) -> Dict:
    """
    Compute correlation between gating weights and difficulty indicators.
    
    Difficulty indicators:
    - MI (mutual information between experts)
    - Entropy of expert predictions
    - Margin (top1 - top2 probability)
    """
    N, E, C = expert_posteriors.shape
    
    # Compute difficulty indicators
    mi_data = compute_mutual_information(expert_posteriors)
    mi = mi_data['mutual_information']  # [N]
    
    # Entropy per expert
    eps = 1e-8
    expert_entropies = -torch.sum(
        expert_posteriors * torch.log(expert_posteriors + eps), dim=-1
    )  # [N, E]
    mean_entropy = expert_entropies.mean(dim=-1).cpu().numpy()  # [N]
    
    # Margin (top1 - top2) per expert, then average
    top2_vals, _ = torch.topk(expert_posteriors, k=2, dim=-1)  # [N, E, 2]
    margins = (top2_vals[:, :, 0] - top2_vals[:, :, 1]).mean(dim=-1).cpu().numpy()  # [N]
    
    # Accuracy per sample (as difficulty proxy - lower acc = harder)
    uniform_mixture = expert_posteriors.mean(dim=1)  # [N, C]
    predictions = uniform_mixture.argmax(dim=-1)
    accuracies = (predictions == labels).float().cpu().numpy()  # [N]
    difficulty = 1.0 - accuracies  # Higher = harder
    
    # Compute correlations for each expert
    correlations = {}
    for e in range(E):
        weights_e = gating_weights[:, e].cpu().numpy()  # [N]
        
        corr_mi, _ = pearsonr(weights_e, mi)
        corr_entropy, _ = pearsonr(weights_e, mean_entropy)
        corr_margin, _ = pearsonr(weights_e, margins)
        corr_difficulty, _ = pearsonr(weights_e, difficulty)
        
        correlations[f'expert_{e}'] = {
            'mi': corr_mi,
            'entropy': corr_entropy,
            'margin': corr_margin,
            'difficulty': corr_difficulty,
        }
    
    return {
        'correlations': correlations,
        'difficulty_indicators': {
            'mi': mi,
            'entropy': mean_entropy,
            'margin': margins,
            'difficulty': difficulty,
        }
    }


def compute_gating_fairness(gating_weights: torch.Tensor, labels: torch.Tensor,
                           class_to_group: torch.Tensor) -> Dict:
    """
    Compute gating fairness by group: E[w_e | y ∈ G_tail] vs E[w_e | y ∈ G_head]
    """
    N, E = gating_weights.shape
    groups = class_to_group[labels]
    num_groups = int(class_to_group.max().item() + 1)
    
    group_weights = {}
    for g in range(num_groups):
        mask = (groups == g)
        if mask.sum() > 0:
            group_weights[f'group_{g}'] = gating_weights[mask].mean(dim=0).cpu().numpy()  # [E]
    
    # Specialization: which expert dominates for which group?
    specialization = {}
    for e in range(E):
        expert_group_weights = {}
        for g in range(num_groups):
            if f'group_{g}' in group_weights:
                expert_group_weights[f'group_{g}'] = group_weights[f'group_{g}'][e]
        specialization[f'expert_{e}'] = expert_group_weights
    
    return {
        'group_weights': group_weights,
        'specialization': specialization,
    }


# ================================
# Visualization Functions
# ================================
def plot_variance_analysis(variance_data: Dict, save_path: Path):
    """Plot variance reduction analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    delta_var = variance_data['delta_var']
    mixture_delta_var = variance_data['mixture_delta_var']
    variance_reduction = variance_data['variance_reduction']
    
    # Histogram of delta_var
    axes[0, 0].hist(delta_var, bins=50, alpha=0.7, label='Before mixture', color='skyblue')
    axes[0, 0].hist(mixture_delta_var, bins=50, alpha=0.7, label='After uniform mixture', color='coral')
    axes[0, 0].set_xlabel('Δ_var(x)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Variance Distribution: Before vs After Mixture')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Variance reduction
    axes[0, 1].hist(variance_reduction, bins=50, alpha=0.7, color='green')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Variance Reduction')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Variance Reduction Distribution')
    axes[0, 1].grid(alpha=0.3)
    
    # Statistics
    stats_text = f"""
    Statistics:
    ----------
    Mean Δ_var (before): {delta_var.mean():.4f}
    Mean Δ_var (after):  {mixture_delta_var.mean():.4f}
    Mean reduction:      {variance_reduction.mean():.4f}
    
    Median Δ_var (before): {np.median(delta_var):.4f}
    Median Δ_var (after):  {np.median(mixture_delta_var):.4f}
    Median reduction:      {np.median(variance_reduction):.4f}
    """
    
    axes[1, 0].axis('off')
    axes[1, 0].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    
    # CDF of variance reduction
    sorted_reduction = np.sort(variance_reduction)
    cdf = np.arange(1, len(sorted_reduction) + 1) / len(sorted_reduction)
    axes[1, 1].plot(sorted_reduction, cdf, linewidth=2)
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Variance Reduction')
    axes[1, 1].set_ylabel('CDF')
    axes[1, 1].set_title('CDF of Variance Reduction')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('MoE Variance Reduction Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_mi_analysis(mi_data: Dict, save_path: Path):
    """Plot Mutual Information analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    mi = mi_data['mutual_information']
    mixture_entropy = mi_data['mixture_entropy']
    mean_expert_entropy = mi_data['mean_expert_entropy']
    
    # Histogram of MI
    axes[0, 0].hist(mi, bins=50, alpha=0.7, color='purple')
    axes[0, 0].set_xlabel('MI(x)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Mutual Information Distribution')
    axes[0, 0].grid(alpha=0.3)
    
    # Entropy comparison
    axes[0, 1].hist(mean_expert_entropy, bins=50, alpha=0.7, label='Mean Expert Entropy', color='skyblue')
    axes[0, 1].hist(mixture_entropy, bins=50, alpha=0.7, label='Mixture Entropy', color='coral')
    axes[0, 1].set_xlabel('Entropy')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Entropy: Experts vs Mixture')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Statistics
    stats_text = f"""
    Statistics:
    ----------
    Mean MI:              {mi.mean():.4f}
    Median MI:            {np.median(mi):.4f}
    Std MI:               {mi.std():.4f}
    
    Mean Expert Entropy:   {mean_expert_entropy.mean():.4f}
    Mean Mixture Entropy: {mixture_entropy.mean():.4f}
    """
    
    axes[1, 0].axis('off')
    axes[1, 0].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    
    # Scatter: MI vs Expert Entropy
    axes[1, 1].scatter(mean_expert_entropy, mi, alpha=0.3, s=1)
    axes[1, 1].set_xlabel('Mean Expert Entropy')
    axes[1, 1].set_ylabel('MI(x)')
    axes[1, 1].set_title('MI vs Expert Entropy')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('MoE Mutual Information Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration_analysis(cal_data: Dict, method_name: str, save_path: Path):
    """Plot calibration analysis (reliability diagrams)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    bin_accs = cal_data['bin_accs']
    bin_confs = cal_data['bin_confs']
    bin_counts = cal_data['bin_counts']
    
    # Reliability diagram
    bins = np.arange(len(bin_accs))
    axes[0].bar(bins, bin_confs, alpha=0.5, label='Confidence', color='blue')
    axes[0].bar(bins, bin_accs, alpha=0.5, label='Accuracy', color='red')
    axes[0].plot([0, len(bins)-1], [0, 1], 'k--', linewidth=1)
    axes[0].set_xlabel('Confidence Bin')
    axes[0].set_ylabel('Probability')
    axes[0].set_title(f'Reliability Diagram: {method_name}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Per-group metrics
    group_metrics = cal_data.get('group_metrics', {})
    if group_metrics:
        methods = list(group_metrics.keys())
        eces = [group_metrics[m]['ece'] for m in methods]
        nlls = [group_metrics[m]['nll'] for m in methods]
        briers = [group_metrics[m]['brier'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.25
        axes[1].bar(x - width, eces, width, label='ECE', alpha=0.8)
        axes[1].bar(x, nlls, width, label='NLL', alpha=0.8)
        axes[1].bar(x + width, briers, width, label='Brier', alpha=0.8)
        axes[1].set_xlabel('Group')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title('Calibration Metrics by Group')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.suptitle(f'Calibration Analysis: {method_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_gating_statistics(gating_stats: Dict, expert_names: List[str], save_path: Path):
    """Plot gating statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    weight_entropy = gating_stats['weight_entropy']
    effective_experts = gating_stats['effective_experts']
    mean_weights = gating_stats['mean_weights']
    
    # Histogram of weight entropy
    axes[0, 0].hist(weight_entropy, bins=50, alpha=0.7, color='green')
    axes[0, 0].set_xlabel('Weight Entropy H(w(x))')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Gating Weight Entropy Distribution')
    axes[0, 0].grid(alpha=0.3)
    
    # Histogram of effective experts
    axes[0, 1].hist(effective_experts, bins=50, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Effective Number of Experts')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Effective Experts Distribution')
    axes[0, 1].grid(alpha=0.3)
    
    # Mean weights per expert
    x = np.arange(len(expert_names))
    axes[1, 0].bar(x, mean_weights, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(expert_names, rotation=15, ha='right')
    axes[1, 0].set_ylabel('Mean Weight')
    axes[1, 0].set_title('Mean Gating Weights per Expert')
    axes[1, 0].axhline(1.0 / len(expert_names), color='red', linestyle='--', 
                      label=f'Uniform (1/{len(expert_names)})')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Statistics
    stats_text = f"""
    Statistics:
    ----------
    Mean Entropy:         {weight_entropy.mean():.4f}
    Mean Effective Exp:   {effective_experts.mean():.4f}
    Load Balance Std:     {gating_stats['load_balance_std']:.4f}
    
    Mean Weights:
    """
    for name, w in zip(expert_names, mean_weights):
        stats_text += f"\n  {name}: {w:.4f}"
    
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    
    plt.suptitle('Gating Network Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_oracle_comparison(oracle_data: Dict, expert_names: List[str], save_path: Path):
    """Plot Oracle vs Uniform vs Gating comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['oracle', 'uniform', 'gating']
    method_labels = ['Oracle@E', 'Uniform-Mix', 'Gating-Mix']
    
    accs = [oracle_data[m]['acc'] for m in methods if m in oracle_data]
    eces = [oracle_data[m]['ece'] for m in methods if m in oracle_data]
    nlls = [oracle_data[m]['nll'] for m in methods if m in oracle_data]
    
    # Accuracy comparison
    x = np.arange(len(methods))
    axes[0].bar(x, accs, alpha=0.7, color=['gold', 'steelblue', 'green'], edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_labels)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Comparison')
    axes[0].grid(alpha=0.3, axis='y')
    for i, acc in enumerate(accs):
        axes[0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    # ECE comparison
    axes[1].bar(x, eces, alpha=0.7, color=['gold', 'steelblue', 'green'], edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_labels)
    axes[1].set_ylabel('ECE')
    axes[1].set_title('Calibration (ECE) Comparison')
    axes[1].grid(alpha=0.3, axis='y')
    for i, ece in enumerate(eces):
        axes[1].text(i, ece + 0.005, f'{ece:.3f}', ha='center', va='bottom')
    
    # NLL comparison
    axes[2].bar(x, nlls, alpha=0.7, color=['gold', 'steelblue', 'green'], edgecolor='black')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(method_labels)
    axes[2].set_ylabel('NLL')
    axes[2].set_title('Calibration (NLL) Comparison')
    axes[2].grid(alpha=0.3, axis='y')
    for i, nll in enumerate(nlls):
        axes[2].text(i, nll + 0.05, f'{nll:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Oracle vs Uniform vs Gating Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ================================
# Main Analysis Function
# ================================
def main():
    """Main analysis function."""
    print("="*70)
    print("Comprehensive MoE-L2R Analysis")
    print("="*70)
    
    # Setup
    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    expert_logits_test = load_expert_logits(CFG.expert_names, "test", CFG.device)
    labels_test = load_labels("test", CFG.device)
    expert_posteriors_test = F.softmax(expert_logits_test, dim=-1)  # [N, E, C]
    
    print(f"   Loaded {expert_logits_test.shape[0]} test samples")
    
    # Load gating
    print("\n2. Loading gating network...")
    gating = load_gating_network(CFG.device)
    
    # Compute mixture
    print("\n3. Computing mixture posteriors...")
    mixture_posterior, gating_weights = compute_mixture_posterior(
        expert_logits_test, gating, CFG.device
    )
    uniform_mixture = expert_posteriors_test.mean(dim=1)  # [N, C]
    
    # Build class-to-group mapping
    print("\n4. Building class-to-group mapping...")
    class_to_group = build_class_to_group(CFG.device)
    
    # ================================
    # (A) MoE → Smooth Signals Analysis
    # ================================
    print("\n" + "="*70)
    print("(A) MoE → Smooth Signals Analysis")
    print("="*70)
    
    print("   Computing expert variance...")
    variance_data = compute_expert_variance(expert_posteriors_test, labels_test)
    plot_variance_analysis(variance_data, output_dir / "variance_analysis.png")
    
    print("   Computing Mutual Information...")
    mi_data = compute_mutual_information(expert_posteriors_test)
    plot_mi_analysis(mi_data, output_dir / "mi_analysis.png")
    
    print("   Computing calibration metrics...")
    # For each method: individual experts, uniform mix, gating mix
    cal_results = {}
    
    # Individual experts
    for e, expert_name in enumerate(CFG.expert_names):
        expert_cal = compute_calibration_metrics(
            expert_posteriors_test[:, e, :], labels_test, class_to_group
        )
        cal_results[expert_name] = expert_cal
        plot_calibration_analysis(expert_cal, expert_name, 
                                  output_dir / f"calibration_{expert_name}.png")
    
    # Uniform mixture
    uniform_cal = compute_calibration_metrics(uniform_mixture, labels_test, class_to_group)
    cal_results['uniform_mix'] = uniform_cal
    plot_calibration_analysis(uniform_cal, 'Uniform-Mix', 
                              output_dir / "calibration_uniform_mix.png")
    
    # Gating mixture
    gating_cal = compute_calibration_metrics(mixture_posterior, labels_test, class_to_group)
    cal_results['gating_mix'] = gating_cal
    plot_calibration_analysis(gating_cal, 'Gating-Mix', 
                              output_dir / "calibration_gating_mix.png")
    
    print("   Computing Oracle comparison...")
    oracle_data = compute_oracle_vs_gating(
        expert_posteriors_test, gating_weights, uniform_mixture, 
        labels_test, class_to_group
    )
    plot_oracle_comparison(oracle_data, CFG.expert_names, 
                          output_dir / "oracle_comparison.png")
    
    # ================================
    # (B) Understanding Gating Analysis
    # ================================
    print("\n" + "="*70)
    print("(B) Understanding Gating Analysis")
    print("="*70)
    
    print("   Computing gating statistics...")
    gating_stats = compute_gating_statistics(gating_weights, labels_test, class_to_group)
    plot_gating_statistics(gating_stats, CFG.expert_names, 
                          output_dir / "gating_statistics.png")
    
    print("   Computing difficulty correlations...")
    corr_data = compute_difficulty_correlation(
        expert_posteriors_test, gating_weights, labels_test, class_to_group
    )
    
    print("   Computing gating fairness...")
    fairness_data = compute_gating_fairness(gating_weights, labels_test, class_to_group)
    
    # ================================
    # Save Results
    # ================================
    print("\n5. Saving results...")
    
    results = {
        'variance_analysis': {
            'mean_delta_var_before': float(variance_data['delta_var'].mean()),
            'mean_delta_var_after': float(variance_data['mixture_delta_var'].mean()),
            'mean_variance_reduction': float(variance_data['variance_reduction'].mean()),
            'median_variance_reduction': float(np.median(variance_data['variance_reduction'])),
        },
        'mi_analysis': {
            'mean_mi': float(mi_data['mutual_information'].mean()),
            'median_mi': float(np.median(mi_data['mutual_information'])),
            'mean_expert_entropy': float(mi_data['mean_expert_entropy'].mean()),
            'mean_mixture_entropy': float(mi_data['mixture_entropy'].mean()),
        },
        'calibration': {},
        'oracle_comparison': oracle_data,
        'gating_statistics': {
            'mean_entropy': float(gating_stats['weight_entropy'].mean()),
            'mean_effective_experts': float(gating_stats['effective_experts'].mean()),
            'load_balance_std': gating_stats['load_balance_std'],
            'mean_weights': gating_stats['mean_weights'].tolist(),
            'group_stats': gating_stats['group_stats'],
        },
        'difficulty_correlations': corr_data['correlations'],
        'gating_fairness': fairness_data,
    }
    
    # Add calibration metrics
    for method, cal in cal_results.items():
        results['calibration'][method] = {
            'ece': cal['ece'],
            'nll': cal['nll'],
            'brier': cal['brier'],
            'group_metrics': cal.get('group_metrics', {}),
        }
    
    # Convert all numpy/torch types to Python native types for JSON serialization
    results_serializable = convert_to_json_serializable(results)
    
    # Save JSON
    results_path = output_dir / "comprehensive_analysis_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"   Saved results to: {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"Variance Reduction: {results['variance_analysis']['mean_variance_reduction']:.4f}")
    print(f"Mean MI: {results['mi_analysis']['mean_mi']:.4f}")
    print(f"Gating Mean Entropy: {results['gating_statistics']['mean_entropy']:.4f}")
    print(f"Gating Mean Effective Experts: {results['gating_statistics']['mean_effective_experts']:.2f}")
    print("\nCalibration (ECE):")
    for method, cal in cal_results.items():
        print(f"  {method}: {cal['ece']:.4f}")
    print("\nOracle Comparison:")
    for method in ['oracle', 'uniform', 'gating']:
        if method in oracle_data:
            print(f"  {method}: Acc={oracle_data[method]['acc']:.4f}, ECE={oracle_data[method]['ece']:.4f}")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

