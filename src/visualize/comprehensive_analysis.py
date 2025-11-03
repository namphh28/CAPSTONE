"""
Comprehensive Analysis & Visualization
====================================

Chứng minh method đúng lý thuyết thông qua:
1. Routing patterns (per-class, per-group)
2. Load balancing & expert utilization
3. Expert diversity & disagreement
4. Ensemble benefits (single vs mixture)
5. Calibration analysis (ECE/Brier)
6. RC curves với ablation studies

Usage:
    python -m src.visualize.comprehensive_analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from src.models.gating_network_map import GatingNetwork


# ============================================================================
# 1. ROUTING PATTERN ANALYSIS
# ============================================================================


def analyze_routing_patterns(
    posteriors: torch.Tensor,
    gating: GatingNetwork,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Phân tích routing patterns theo class và group.

    Returns:
        - weights: [N, E] gating weights
        - per_class_usage: {class_id: [E] mean weights}
        - per_group_usage: {group: [E] mean weights}
        - effective_experts: [N] number of active experts per sample
    """
    with torch.no_grad():
        # Get gating weights
        weights, _ = gating(posteriors)  # [N, E]
        weights_np = weights.cpu().numpy()
        labels_np = labels.cpu().numpy()

        num_classes = len(np.unique(labels_np))
        num_experts = weights.shape[1]

        # Per-class usage
        per_class_usage = {}
        for c in range(num_classes):
            class_mask = labels_np == c
            if class_mask.sum() > 0:
                per_class_usage[c] = weights_np[class_mask].mean(axis=0)

        # Effective experts (entropy-based)
        entropies = -np.sum(weights_np * np.log(weights_np + 1e-8), axis=1)
        effective_experts = np.exp(entropies)

        return {
            "weights": weights_np,
            "per_class_usage": per_class_usage,
            "effective_experts": effective_experts,
        }


def plot_routing_patterns(routing_data: Dict, save_path: Path):
    """Plot routing patterns."""
    fig = plt.figure(figsize=(20, 12))

    # =========================================================================
    # Plot 1: Expert Usage Heatmap (per-class, top 20 classes)
    # =========================================================================
    ax1 = plt.subplot(3, 3, 1)

    per_class = routing_data["per_class_usage"]
    top_classes = sorted(per_class.keys())[:20]
    usage_matrix = np.array([per_class[c] for c in top_classes])

    sns.heatmap(
        usage_matrix,
        ax=ax1,
        cmap="YlOrRd",
        xticklabels=[f"Expert {i + 1}" for i in range(usage_matrix.shape[1])],
        yticklabels=[f"Class {c}" for c in top_classes],
        cbar_kws={"label": "Mean Weight"},
    )
    ax1.set_title("Expert Usage by Class (Top 20)", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Expert", fontsize=10)
    ax1.set_ylabel("Class", fontsize=10)

    # =========================================================================
    # Plot 2: Expert Usage Distribution
    # =========================================================================
    ax2 = plt.subplot(3, 3, 2)

    weights = routing_data["weights"]
    for e in range(weights.shape[1]):
        ax2.hist(
            weights[:, e],
            bins=50,
            alpha=0.6,
            label=f"Expert {e + 1} (mean={weights[:, e].mean():.3f})",
        )

    ax2.set_xlabel("Gating Weight", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.set_title("Expert Weight Distribution", fontweight="bold", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # =========================================================================
    # Plot 3: Effective Experts per Sample
    # =========================================================================
    ax3 = plt.subplot(3, 3, 3)

    effective = routing_data["effective_experts"]
    ax3.hist(effective, bins=50, color="steelblue", alpha=0.7)
    ax3.axvline(
        effective.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {effective.mean():.2f}",
    )
    ax3.set_xlabel("Effective Number of Experts", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.set_title(f"Expert Diversity (Entropy-based)", fontweight="bold", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # =========================================================================
    # Plot 4: Load Balance (Standard deviation of expert usage)
    # =========================================================================
    ax4 = plt.subplot(3, 3, 4)

    expert_usage = weights.mean(axis=0)  # [E] mean usage per expert
    expert_std = weights.std(axis=0)  # [E] std per expert

    x_pos = np.arange(len(expert_usage))
    bars = ax4.bar(
        x_pos,
        expert_usage,
        yerr=expert_std,
        capsize=5,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"Expert {i + 1}" for i in range(len(expert_usage))])
    ax4.set_ylabel("Mean Weight", fontsize=10)
    ax4.set_title(f"Load Balance (Ideal=0.33)", fontweight="bold", fontsize=11)
    ax4.axhline(
        1 / len(expert_usage), color="red", linestyle="--", label="Perfect Balance"
    )
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3, axis="y")

    # Add values on bars
    for bar, val in zip(bars, expert_usage):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # =========================================================================
    # Plot 5: Expert Correlation
    # =========================================================================
    ax5 = plt.subplot(3, 3, 5)

    corr_matrix = np.corrcoef(weights.T)
    sns.heatmap(
        corr_matrix,
        ax=ax5,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=[f"E{i + 1}" for i in range(len(corr_matrix))],
        yticklabels=[f"E{i + 1}" for i in range(len(corr_matrix))],
    )
    ax5.set_title("Expert Correlation Matrix", fontweight="bold", fontsize=11)

    # =========================================================================
    # Plot 6: Usage Sparsity (Distribution of top expert weight)
    # =========================================================================
    ax6 = plt.subplot(3, 3, 6)

    top_weights = weights.max(axis=1)  # [N]
    ax6.hist(top_weights, bins=50, color="darkgreen", alpha=0.7)
    ax6.axvline(
        top_weights.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {top_weights.mean():.3f}",
    )
    ax6.set_xlabel("Top Expert Weight", fontsize=10)
    ax6.set_ylabel("Frequency", fontsize=10)
    ax6.set_title(
        "Routing Sparsity (1=fully concentrated)", fontweight="bold", fontsize=11
    )
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)

    # =========================================================================
    # Plot 7-9: Per-Class Usage (Head vs Tail)
    # =========================================================================
    per_class = routing_data["per_class_usage"]

    # Head classes (0-68)
    ax7 = plt.subplot(3, 3, 7)
    head_classes = [c for c in sorted(per_class.keys()) if c < 69]
    if head_classes:
        head_usage = np.array([per_class[c] for c in head_classes])
        ax7.bar(
            range(len(head_classes)),
            head_usage.mean(axis=1),
            color="skyblue",
            alpha=0.8,
            edgecolor="black",
        )
        ax7.set_xticks(range(0, len(head_classes), 10))
        ax7.set_xticklabels([head_classes[i] for i in range(0, len(head_classes), 10)])
        ax7.set_ylabel("Mean Weight", fontsize=10)
        ax7.set_title(
            f"Head Classes Usage (Classes 0-68)", fontweight="bold", fontsize=11
        )
        ax7.grid(alpha=0.3, axis="y")

    # Tail classes (69-99)
    ax8 = plt.subplot(3, 3, 8)
    tail_classes = [c for c in sorted(per_class.keys()) if c >= 69]
    if tail_classes:
        tail_usage = np.array([per_class[c] for c in tail_classes])
        ax8.bar(
            range(len(tail_classes)),
            tail_usage.mean(axis=1),
            color="coral",
            alpha=0.8,
            edgecolor="black",
        )
        ax8.set_xticks(range(0, len(tail_classes), 5))
        ax8.set_xticklabels([tail_classes[i] for i in range(0, len(tail_classes), 5)])
        ax8.set_ylabel("Mean Weight", fontsize=10)
        ax8.set_title(
            f"Tail Classes Usage (Classes 69-99)", fontweight="bold", fontsize=11
        )
        ax8.grid(alpha=0.3, axis="y")

    # Head vs Tail Comparison
    ax9 = plt.subplot(3, 3, 9)
    if head_classes and tail_classes:
        head_mean = [per_class[c].mean() for c in head_classes]
        tail_mean = [per_class[c].mean() for c in tail_classes]

        ax9.bar(
            ["Head (0-68)", "Tail (69-99)"],
            [np.mean(head_mean), np.mean(tail_mean)],
            color=["skyblue", "coral"],
            alpha=0.8,
            edgecolor="black",
        )
        ax9.set_ylabel("Mean Expert Usage", fontsize=10)
        ax9.set_title("Head vs Tail Expert Utilization", fontweight="bold", fontsize=11)
        ax9.grid(alpha=0.3, axis="y")

    plt.suptitle("Routing Pattern Analysis", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved routing patterns to {save_path}")
    plt.close()


# ============================================================================
# 2. EXPERT DISAGREEMENT & DIVERSITY
# ============================================================================


def analyze_expert_disagreement(posteriors: torch.Tensor, labels: torch.Tensor) -> Dict:
    """
    Phân tích disagreement giữa các experts.
    """
    with torch.no_grad():
        # Get predictions
        predictions = posteriors.argmax(dim=-1)  # [N, E]

        # Disagreement: fraction of experts that agree with majority
        majority_pred, _ = torch.mode(predictions, dim=1)  # [N]
        agreement = (predictions == majority_pred.unsqueeze(1)).float()  # [N, E]
        disagreement = 1 - agreement.mean(dim=1)  # [N]

        # Per-class disagreement
        per_class_disagreement = {}
        for c in range(100):
            mask = labels == c
            if mask.sum() > 0:
                per_class_disagreement[c] = disagreement[mask].mean().item()

        return {
            "disagreement": disagreement.numpy(),
            "per_class_disagreement": per_class_disagreement,
            "mean_disagreement": disagreement.mean().item(),
        }


def plot_expert_disagreement(
    disagreement_data: Dict, posteriors: torch.Tensor, save_path: Path
):
    """Plot expert disagreement analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Overall disagreement distribution
    ax1 = axes[0, 0]
    ax1.hist(disagreement_data["disagreement"], bins=50, color="steelblue", alpha=0.7)
    ax1.axvline(
        disagreement_data["mean_disagreement"],
        color="red",
        linestyle="--",
        label=f"Mean: {disagreement_data['mean_disagreement']:.3f}",
    )
    ax1.set_xlabel("Disagreement Rate", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Expert Disagreement Distribution", fontweight="bold", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Plot 2: Per-class disagreement
    ax2 = axes[0, 1]
    per_class = disagreement_data["per_class_disagreement"]
    classes = sorted(per_class.keys())
    disagreements = [per_class[c] for c in classes]

    ax2.plot(classes, disagreements, "o-", linewidth=1.5, markersize=3)
    ax2.axhline(
        np.mean(disagreements),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(disagreements):.3f}",
    )
    ax2.set_xlabel("Class Index", fontsize=11)
    ax2.set_ylabel("Disagreement Rate", fontsize=11)
    ax2.set_title("Per-Class Disagreement", fontweight="bold", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Plot 3: Head vs Tail disagreement
    ax3 = axes[1, 0]
    head_classes = [c for c in classes if c < 69]
    tail_classes = [c for c in classes if c >= 69]

    head_disagreement = [per_class[c] for c in head_classes]
    tail_disagreement = [per_class[c] for c in tail_classes]

    ax3.boxplot(
        [head_disagreement, tail_disagreement], labels=["Head (0-68)", "Tail (69-99)"]
    )
    ax3.set_ylabel("Disagreement Rate", fontsize=11)
    ax3.set_title("Head vs Tail Expert Disagreement", fontweight="bold", fontsize=12)
    ax3.grid(alpha=0.3, axis="y")

    # Plot 4: Agreement matrix between experts
    ax4 = axes[1, 1]

    # For first 1000 samples, compute agreement
    N_subset = min(1000, len(disagreement_data["disagreement"]))
    pred_subset = posteriors[:N_subset].argmax(dim=-1).cpu().numpy()

    # Compute pairwise agreement
    num_experts = pred_subset.shape[1]
    agreement_matrix = np.zeros((num_experts, num_experts))
    for i in range(num_experts):
        for j in range(num_experts):
            if i != j:
                agreement_matrix[i, j] = (pred_subset[:, i] == pred_subset[:, j]).mean()

    sns.heatmap(
        agreement_matrix,
        ax=ax4,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=[f"E{i + 1}" for i in range(num_experts)],
        yticklabels=[f"E{i + 1}" for i in range(num_experts)],
    )
    ax4.set_title("Expert Pairwise Agreement Matrix", fontweight="bold", fontsize=12)

    plt.suptitle(
        "Expert Disagreement & Diversity Analysis", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved disagreement analysis to {save_path}")
    plt.close()


# ============================================================================
# 3. ENSEMBLE BENEFITS (Single vs Mixture)
# ============================================================================


def compare_single_vs_ensemble(
    posteriors: torch.Tensor,
    gating: GatingNetwork,
    labels: torch.Tensor,
    save_path: Path,
):
    """So sánh performance của từng expert vs mixture."""
    with torch.no_grad():
        # Get mixture predictions
        weights, _ = gating(posteriors)
        mixture = gating.get_mixture_posterior(posteriors, weights)
        mixture_pred = mixture.argmax(dim=-1)

        # Get individual expert predictions
        expert_preds = posteriors.argmax(dim=-1)  # [N, E]

        # Compute accuracies
        mixture_acc = (mixture_pred == labels).float().mean()
        expert_accs = []
        for e in range(expert_preds.shape[1]):
            expert_accs.append((expert_preds[:, e] == labels).float().mean())

        # Confusion matrices (first 20 classes)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        expert_names = ["CE", "LogitAdjust", "BalancedSoftmax"]
        for e in range(3):
            ax = axes[0, e]
            pred_e = expert_preds[:, e].cpu().numpy()
            labels_e = labels.cpu().numpy()

            # Subsample for visualization
            N_viz = min(5000, len(labels_e))
            pred_viz = pred_e[:N_viz]
            labels_viz = labels_e[:N_viz]

            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(labels_viz, pred_viz, labels=range(20))
            sns.heatmap(
                cm,
                ax=ax,
                annot=False,
                fmt="d",
                cmap="Blues",
                cbar_kws={"label": "Count"},
            )
            ax.set_title(
                f"{expert_names[e]} (Acc: {expert_accs[e]:.3f})", fontweight="bold"
            )
            ax.set_xlabel("Predicted (First 20 classes)", fontsize=9)
            ax.set_ylabel("True Class", fontsize=9)

        # Mixture confusion
        ax_mix = axes[1, 1]
        mixture_pred_viz = mixture_pred[:N_viz].cpu().numpy()
        cm_mix = confusion_matrix(labels_viz, mixture_pred_viz, labels=range(20))
        sns.heatmap(
            cm_mix,
            ax=ax_mix,
            annot=False,
            fmt="d",
            cmap="Blues",
            cbar_kws={"label": "Count"},
        )
        ax_mix.set_title(f"Mixture (Acc: {mixture_acc:.3f})", fontweight="bold")
        ax_mix.set_xlabel("Predicted (First 20 classes)", fontsize=9)
        ax_mix.set_ylabel("True Class", fontsize=9)

        # Accuracy comparison
        ax_acc = axes[1, 0]
        methods = expert_names + ["Mixture"]
        accuracies = expert_accs + [mixture_acc.item()]
        colors = ["steelblue"] * 3 + ["green"]

        bars = ax_acc.bar(
            methods,
            accuracies,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax_acc.set_ylabel("Accuracy", fontsize=11)
        ax_acc.set_title("Single Expert vs Mixture", fontweight="bold", fontsize=12)
        ax_acc.set_ylim([0, 1.0])
        ax_acc.grid(alpha=0.3, axis="y")

        # Add values on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax_acc.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Improvement
        best_single = max(expert_accs)
        improvement = mixture_acc - best_single
        ax_acc.text(
            0.5,
            0.95,
            f"Improvement: +{improvement:.3f}",
            transform=ax_acc.transAxes,
            ha="center",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
        )

        # Per-class accuracy
        ax_pc = axes[1, 2]
        class_accs = []
        for c in range(20):
            mask = labels == c
            if mask.sum() > 0:
                class_accs.append(
                    (mixture_pred[mask] == labels[mask]).float().mean().item()
                )
            else:
                class_accs.append(0.0)

        ax_pc.plot(range(20), class_accs, "o-", linewidth=1.5, markersize=4)
        ax_pc.set_xlabel("Class Index (First 20)", fontsize=10)
        ax_pc.set_ylabel("Accuracy", fontsize=10)
        ax_pc.set_title("Mixture Per-Class Accuracy", fontweight="bold", fontsize=12)
        ax_pc.grid(alpha=0.3)

        plt.suptitle(
            "Ensemble Benefits: Single Experts vs Mixture",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ensemble comparison to {save_path}")
        plt.close()


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================


def run_comprehensive_analysis():
    """Run all analyses."""
    print("=" * 70)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    logits_dir = Path("./outputs/logits/cifar100_lt_if100")
    splits_dir = Path("./data/cifar100_lt_if100_splits_fixed")

    expert_names = ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"]

    # Load expert logits
    logits_list = []
    for expert_name in expert_names:
        logits_path = logits_dir / expert_name / "test_logits.pt"
        logits_e = torch.load(logits_path, map_location="cpu").float()
        logits_list.append(logits_e)

    logits = torch.stack(logits_list, dim=0).transpose(0, 1)  # [N, E, C]
    posteriors = torch.softmax(logits, dim=-1)

    print(f"  Loaded: {logits.shape[0]} samples, {logits.shape[1]} experts")

    # Load labels
    import torchvision

    dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=False)
    with open(splits_dir / "test_indices.json", "r") as f:
        indices = json.load(f)
    labels = torch.tensor([dataset.targets[i] for i in indices])
    print(f"[OK] Loaded {len(labels)} labels")

    # Load gating model
    print("\n2. Loading gating model...")
    gating_path = Path("./checkpoints/gating_map/cifar100_lt_if100/final_gating.pth")
    checkpoint = torch.load(gating_path, map_location="cpu")

    gating = GatingNetwork(num_experts=3, num_classes=100, routing="dense")
    gating.load_state_dict(checkpoint["model_state_dict"])
    gating.eval()
    print("[OK] Gating model loaded")

    # Create output directory
    output_dir = Path("./results/comprehensive_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analyses
    print("\n3. Analyzing routing patterns...")
    routing_data = analyze_routing_patterns(posteriors, gating, labels)
    plot_routing_patterns(routing_data, output_dir / "routing_patterns.png")

    print("\n4. Analyzing expert disagreement...")
    disagreement_data = analyze_expert_disagreement(posteriors, labels)
    plot_expert_disagreement(
        disagreement_data, posteriors, output_dir / "expert_disagreement.png"
    )

    print("\n5. Comparing single vs ensemble...")
    compare_single_vs_ensemble(
        posteriors, gating, labels, output_dir / "ensemble_comparison.png"
    )

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETED!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_analysis()
