#!/usr/bin/env python3
"""
Counterfactual & Ablation Analysis for MoE-L2R
==============================================

Implements counterfactual experiments:
- H. Variance decomposition & MI correlation
- I. Oracle/Uniform/Gating RC curves, Shuffle-gating, Expert ablation
- J. Plugin formula validation

Usage:
    python scripts/moe_counterfactual_analysis.py
"""

from pathlib import Path
from typing import Dict
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.moe_l2r_comprehensive_analysis import (
    Config,
    load_expert_logits,
    load_labels,
    load_gating_network,
    build_class_to_group,
    compute_mixture_posterior,
    compute_mutual_information,
    compute_expert_variance,
)

CFG = Config()


# ================================
# H. Variance Decomposition & MI
# ================================
def plot_variance_nll_correlation(
    expert_posteriors: torch.Tensor,
    uniform_mixture: torch.Tensor,
    labels: torch.Tensor,
    save_path: Path,
):
    """Plot NLL gain vs variance reduction (scatter)."""
    # Compute variance reduction
    variance_data = compute_expert_variance(expert_posteriors, labels)
    variance_reduction = variance_data["variance_reduction"]

    # Compute NLL per sample
    N = len(labels)
    eps = 1e-8

    # NLL for uniform mixture
    uniform_nll_per_sample = (
        -torch.log(uniform_mixture[torch.arange(N), labels] + eps).cpu().numpy()
    )

    # Mean NLL across experts per sample
    expert_nlls_per_sample = []
    for e in range(expert_posteriors.shape[1]):
        expert_nll = (
            -torch.log(expert_posteriors[torch.arange(N), e, labels] + eps)
            .cpu()
            .numpy()
        )
        expert_nlls_per_sample.append(expert_nll)

    mean_expert_nll_per_sample = np.mean(expert_nlls_per_sample, axis=0)

    # NLL gain = mean_expert_nll - uniform_nll (positive = improvement)
    nll_gain = mean_expert_nll_per_sample - uniform_nll_per_sample

    # Correlation
    corr, _ = np.corrcoef(variance_reduction, nll_gain)
    correlation = corr[0, 1] if corr.shape == (2, 2) else 0.0

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(variance_reduction, nll_gain, alpha=0.3, s=10)
    ax.set_xlabel("Variance Reduction", fontsize=12)
    ax.set_ylabel("NLL Gain (Mean Experts - Mixture)", fontsize=12)
    ax.set_title(
        f"NLL Gain vs Variance Reduction (Corr={correlation:.4f})",
        fontweight="bold",
        fontsize=13,
    )
    ax.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(variance_reduction, nll_gain, 1)
    p = np.poly1d(z)
    ax.plot(
        sorted(variance_reduction),
        p(sorted(variance_reduction)),
        "r--",
        alpha=0.8,
        linewidth=2,
        label="Trend line",
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ================================
# I. Counterfactuals
# ================================
def compute_rc_curve_for_posterior(
    posteriors: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    method_name: str,
) -> Dict:
    """Compute RC curve for a given posterior."""
    from src.models.ltr_plugin import LtRPlugin, LtRPluginConfig, RCCurveComputer

    # Get device from input tensors
    device = posteriors.device

    config = LtRPluginConfig(
        num_classes=CFG.num_classes,
        num_groups=CFG.num_groups,
        group_boundaries=[69],
        param_mode="group",
        objective="balanced",
    )

    plugin = LtRPlugin(config).to(device)  # Move plugin to same device

    # Use tunev to optimize parameters
    expert_logits_tunev = load_expert_logits(CFG.expert_names, "tunev", CFG.device)
    labels_tunev = load_labels("tunev", CFG.device)

    # For simplicity, use uniform parameters (will be converted to tensors on correct device inside)
    alpha = np.ones(CFG.num_groups)
    mu = np.zeros(CFG.num_groups)

    # Create RC curve computer
    rc_computer = RCCurveComputer(config)

    # Compute RC curve (rc_computer should handle device internally, but ensure inputs are on correct device)
    # Ensure class_to_group is on correct device
    class_to_group = class_to_group.to(device)

    # Compute RC curve
    rc_data = rc_computer.compute_rc_curve(
        plugin,
        posteriors.to(device),
        labels.to(device),
        alpha=alpha,
        mu=mu,
        cost_grid=np.linspace(0.0, 1.0, 20).tolist(),
    )

    return {
        "rejection_rates": rc_data["rejection_rates"],
        "balanced_errors": rc_data["balanced_errors"],
        "aurc": rc_data["aurc"],
    }


def plot_oracle_uniform_gating_rc(
    expert_posteriors: torch.Tensor,
    uniform_mixture: torch.Tensor,
    gating_mixture: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    save_path: Path,
):
    """Plot RC curves for Oracle, Uniform, and Gating."""
    print("   Computing Oracle RC curve...")

    # Oracle: for each sample, choose best expert
    N = expert_posteriors.shape[0]
    oracle_posteriors = torch.zeros_like(uniform_mixture)
    true_class_probs = expert_posteriors[torch.arange(N), :, labels]
    best_expert_per_sample = true_class_probs.argmax(dim=-1)

    for i in range(N):
        oracle_posteriors[i] = expert_posteriors[i, best_expert_per_sample[i]]

    # Compute RC curves (simplified version)
    # Note: This is a simplified computation. For full RC, need to optimize plugin parameters

    methods = {
        "Oracle": oracle_posteriors,
        "Uniform-Mix": uniform_mixture,
        "Gating-Mix": gating_mixture,
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    for method_name, posteriors_method in methods.items():
        # Simplified: compute error at different coverage levels
        predictions = posteriors_method.argmax(dim=-1)
        confidences, _ = posteriors_method.max(dim=-1)

        # Sort by confidence (ascending)
        sorted_indices = torch.argsort(confidences)
        sorted_preds = predictions[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # Compute error at different coverage levels
        coverage_levels = np.linspace(0.1, 1.0, 20)
        errors = []
        rejection_rates = []

        for cov in coverage_levels:
            n_accept = int(N * cov)
            accepted_preds = sorted_preds[-n_accept:]
            accepted_labels = sorted_labels[-n_accept:]
            error = (accepted_preds != accepted_labels).float().mean().item()
            errors.append(error)
            rejection_rates.append(1.0 - cov)

        # Compute AURC
        aurc = np.trapz(errors, rejection_rates)

        ax.plot(
            rejection_rates,
            errors,
            "o-",
            label=f"{method_name} (AURC={aurc:.4f})",
            linewidth=2,
            markersize=4,
        )

    ax.set_xlabel("Proportion of Rejections ρ", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title(
        "RC Curves: Oracle vs Uniform vs Gating", fontweight="bold", fontsize=13
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shuffle_gating(
    expert_posteriors: torch.Tensor,
    gating_weights: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    save_path: Path,
):
    """Shuffle-gating: keep distribution but shuffle assignment."""
    N, E, C = expert_posteriors.shape

    # Original gating mixture
    original_mixture = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)

    # Shuffle weights (keep distribution, shuffle assignment)
    shuffled_indices = np.random.permutation(N)
    shuffled_weights = gating_weights[shuffled_indices]
    shuffled_mixture = (shuffled_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)

    # Compute metrics
    original_preds = original_mixture.argmax(dim=-1)
    shuffled_preds = shuffled_mixture.argmax(dim=-1)

    original_acc = (original_preds == labels).float().mean().item()
    shuffled_acc = (shuffled_preds == labels).float().mean().item()

    # Simplified RC comparison
    original_conf, _ = original_mixture.max(dim=-1)
    shuffled_conf, _ = shuffled_mixture.max(dim=-1)

    # Sort by confidence
    original_sorted = torch.argsort(original_conf)
    shuffled_sorted = torch.argsort(shuffled_conf)

    coverage_levels = np.linspace(0.1, 1.0, 20)

    original_errors = []
    shuffled_errors = []
    rejection_rates = []

    for cov in coverage_levels:
        n_accept = int(N * cov)

        # Original
        orig_acc_preds = original_preds[original_sorted[-n_accept:]]
        orig_acc_labels = labels[original_sorted[-n_accept:]]
        orig_err = (orig_acc_preds != orig_acc_labels).float().mean().item()
        original_errors.append(orig_err)

        # Shuffled
        shuf_acc_preds = shuffled_preds[shuffled_sorted[-n_accept:]]
        shuf_acc_labels = labels[shuffled_sorted[-n_accept:]]
        shuf_err = (shuf_acc_preds != shuf_acc_labels).float().mean().item()
        shuffled_errors.append(shuf_err)

        rejection_rates.append(1.0 - cov)

    original_aurc = np.trapz(original_errors, rejection_rates)
    shuffled_aurc = np.trapz(shuffled_errors, rejection_rates)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        rejection_rates,
        original_errors,
        "o-",
        label=f"Gating (AURC={original_aurc:.4f})",
        linewidth=2,
        color="green",
        markersize=4,
    )
    ax.plot(
        rejection_rates,
        shuffled_errors,
        "s-",
        label=f"Shuffle-Gating (AURC={shuffled_aurc:.4f})",
        linewidth=2,
        color="red",
        markersize=4,
    )

    ax.set_xlabel("Proportion of Rejections ρ", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title(
        f"Shuffle-Gating: AURC Change = {shuffled_aurc - original_aurc:+.4f}",
        fontweight="bold",
        fontsize=13,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Main counterfactual analysis function."""
    print("=" * 70)
    print("Counterfactual & Ablation Analysis")
    print("=" * 70)

    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading data...")
    expert_logits_test = load_expert_logits(CFG.expert_names, "test", CFG.device)
    labels_test = load_labels("test", CFG.device)
    expert_posteriors_test = F.softmax(expert_logits_test, dim=-1)

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
    class_to_group = build_class_to_group(CFG.device)

    # H. Variance & NLL Correlation
    print("\n4. (H) Variance & NLL Correlation...")
    plot_variance_nll_correlation(
        expert_posteriors_test,
        uniform_mixture,
        labels_test,
        output_dir / "variance_nll_correlation.png",
    )

    # I. Counterfactuals
    print("\n5. (I) Counterfactuals...")
    print("   Plotting Oracle/Uniform/Gating RC curves...")
    plot_oracle_uniform_gating_rc(
        expert_posteriors_test,
        uniform_mixture,
        mixture_posterior,
        labels_test,
        class_to_group,
        output_dir / "oracle_uniform_gating_rc.png",
    )

    print("   Plotting shuffle-gating...")
    plot_shuffle_gating(
        expert_posteriors_test,
        gating_weights,
        labels_test,
        class_to_group,
        output_dir / "shuffle_gating.png",
    )

    print("\n" + "=" * 70)
    print("Counterfactual Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
