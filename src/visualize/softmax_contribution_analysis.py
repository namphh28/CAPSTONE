"""
Softmax Distribution & Expert Contribution Analysis
===================================================

Visualize:
1. Softmax probability distributions: Single vs Mixture
2. Expert contribution heatmaps per class
3. How mixture "smooths" the distribution
4. Contribution breakdown by class groups

Usage:
    python -m src.visualize.softmax_contribution_analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List
import json

from src.models.gating_network_map import GatingNetwork


def compare_softmax_distributions(
    posteriors: torch.Tensor,
    gating: GatingNetwork,
    labels: torch.Tensor,
    expert_names: List[str],
    save_path: Path,
):
    """
    So sánh softmax distributions giữa single experts và mixture.

    Visualization:
    1. Probability distributions (single vs mixture) for sample images
    2. Entropy comparison (mixture có entropy cao hơn → smoother)
    3. Confidence distribution comparison
    4. Per-expert contribution breakdown
    """

    with torch.no_grad():
        print("Comparing softmax distributions...")

        # Get mixture
        weights, _ = gating(posteriors)
        mixture = gating.get_mixture_posterior(posteriors, weights)

        # Sample some test cases
        N_samples = 20
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(len(labels), N_samples, replace=False)

        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        for idx, i in enumerate(sample_indices):
            if idx >= 6:  # Show first 6 samples only
                break

            ax = fig.add_subplot(2, 3, idx + 1)

            # Get true class
            true_class = labels[i].item()

            # Get distributions
            expert_posteriors = posteriors[i].cpu().numpy()  # [E, C]
            mixture_posterior = mixture[i].cpu().numpy()  # [C]

            # Find top-5 classes for mixture
            top5_indices = mixture_posterior.argsort()[-5:][::-1]
            top5_values = mixture_posterior[top5_indices]

            # Plot
            x = np.arange(len(top5_indices))
            colors = ["red" if c == true_class else "steelblue" for c in top5_indices]

            bars = ax.bar(x, top5_values, color=colors, alpha=0.7, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels([f"C{c}" for c in top5_indices], rotation=45)
            ax.set_ylabel("Probability", fontsize=9)
            ax.set_title(
                f"Sample {i}: True=C{true_class}", fontweight="bold", fontsize=10
            )
            ax.grid(alpha=0.3, axis="y")

            # Add expert contributions for top prediction
            top_pred = top5_indices[0]
            for e_idx, (name, expert_post) in enumerate(
                zip(expert_names, expert_posteriors)
            ):
                expert_prob = expert_post[top_pred]
                ax.text(
                    0,
                    bars[0].get_height() + 0.02,
                    f"{name[0]}:{expert_prob:.2f}",
                    fontsize=7,
                    ha="center",
                )

        plt.suptitle(
            "Softmax Distribution: Top-5 Classes (Sample Images)",
            fontsize=16,
            fontweight="bold",
        )
        plt.savefig(save_path / "softmax_samples.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ========================================================================
        # Plot 2: Confidence Distribution Comparison
        # ========================================================================
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # Get confidences
        expert_confidences = []
        for e in range(posteriors.shape[1]):
            conf = posteriors[:, e].max(dim=-1)[0]  # [N]
            expert_confidences.append(conf.cpu().numpy())

        mixture_confidence = mixture.max(dim=-1)[0].cpu().numpy()

        # Plot 1: Confidence histograms
        ax1 = axes[0, 0]
        for e_idx, (name, conf) in enumerate(zip(expert_names, expert_confidences)):
            ax1.hist(conf, bins=50, alpha=0.5, label=name, histtype="step", linewidth=2)
        ax1.hist(
            mixture_confidence,
            bins=50,
            alpha=0.7,
            label="Mixture",
            color="green",
            histtype="step",
            linewidth=2,
        )
        ax1.set_xlabel("Confidence (Max Probability)", fontsize=11)
        ax1.set_ylabel("Frequency", fontsize=11)
        ax1.set_title(
            "Confidence Distribution Comparison", fontweight="bold", fontsize=12
        )
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3, axis="y")

        # Plot 2: Entropy comparison
        ax2 = axes[0, 1]

        def compute_entropy(probs):
            return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        expert_entropies = []
        for e in range(posteriors.shape[1]):
            ent = compute_entropy(posteriors[:, e])
            expert_entropies.append(ent.cpu().numpy())

        mixture_entropy = compute_entropy(mixture).cpu().numpy()

        for e_idx, (name, ent) in enumerate(zip(expert_names, expert_entropies)):
            ax2.hist(ent, bins=50, alpha=0.5, label=name, histtype="step", linewidth=2)
        ax2.hist(
            mixture_entropy,
            bins=50,
            alpha=0.7,
            label="Mixture",
            color="green",
            histtype="step",
            linewidth=2,
        )
        ax2.set_xlabel("Entropy (bits)", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.set_title(
            "Entropy Distribution (Higher = More Uniform)",
            fontweight="bold",
            fontsize=12,
        )
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3, axis="y")

        # Add mean annotations
        for e_idx, (name, ent) in enumerate(zip(expert_names, expert_entropies)):
            mean_ent = np.mean(ent)
            ax2.axvline(mean_ent, linestyle=":", alpha=0.7, linewidth=1.5)
            ax2.text(
                mean_ent,
                ax2.get_ylim()[1] * 0.9,
                f"{name}={mean_ent:.2f}",
                rotation=90,
                fontsize=8,
                va="top",
            )

        # Plot 3: Confidence vs Entropy scatter
        ax3 = axes[0, 2]

        ax3.scatter(
            expert_confidences[0],
            expert_entropies[0],
            alpha=0.3,
            s=10,
            label=expert_names[0],
        )
        ax3.scatter(
            expert_confidences[1],
            expert_entropies[1],
            alpha=0.3,
            s=10,
            label=expert_names[1],
        )
        ax3.scatter(
            expert_confidences[2],
            expert_entropies[2],
            alpha=0.3,
            s=10,
            label=expert_names[2],
        )
        ax3.scatter(
            mixture_confidence,
            mixture_entropy,
            alpha=0.5,
            s=20,
            label="Mixture",
            color="green",
            marker="x",
        )

        ax3.set_xlabel("Confidence", fontsize=11)
        ax3.set_ylabel("Entropy", fontsize=11)
        ax3.set_title("Confidence vs Entropy", fontweight="bold", fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

        # ========================================================================
        # Plot 4-6: Contribution Analysis
        # ========================================================================

        # Compute contribution matrix
        contribution_matrix = np.zeros((100, 3))  # [class, expert]

        for c in range(100):
            mask = labels == c
            if mask.sum() > 0:
                # Average weights for this class
                class_weights = weights[mask].mean(dim=0)  # [E]
                contribution_matrix[c] = class_weights.cpu().numpy()

        # Plot 4: Contribution heatmap (ALL classes)
        ax4 = axes[1, 0]

        sns.heatmap(
            contribution_matrix.T,
            ax=ax4,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            xticklabels=False,
            yticklabels=expert_names,
            cbar_kws={"label": "Mean Weight"},
        )
        ax4.axvline(68.5, color="red", linestyle="--", linewidth=2)
        ax4.set_xlabel("Class Index", fontsize=11)
        ax4.set_ylabel("Expert", fontsize=11)
        ax4.set_title(
            "Expert Contribution (All 100 Classes)", fontweight="bold", fontsize=12
        )

        # Plot 5: Contribution by class group
        ax5 = axes[1, 1]

        head_weights = weights[labels < 69].mean(dim=0).cpu().numpy()
        tail_weights = weights[labels >= 69].mean(dim=0).cpu().numpy()

        x = np.arange(len(expert_names))
        width = 0.35

        bars1 = ax5.bar(
            x - width / 2,
            head_weights,
            width,
            label="Head (0-68)",
            alpha=0.8,
            edgecolor="black",
            color="skyblue",
        )
        bars2 = ax5.bar(
            x + width / 2,
            tail_weights,
            width,
            label="Tail (69-99)",
            alpha=0.8,
            edgecolor="black",
            color="coral",
        )

        ax5.set_ylabel("Mean Weight", fontsize=11)
        ax5.set_title("Expert Weight by Class Group", fontweight="bold", fontsize=12)
        ax5.set_xticks(x)
        ax5.set_xticklabels(expert_names)
        ax5.legend()
        ax5.grid(alpha=0.3, axis="y")
        ax5.set_ylim([0, 1.0])

        # Add values
        for bars, vals in [(bars1, head_weights), (bars2, tail_weights)]:
            for bar, val in zip(bars, vals):
                ax5.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        # Plot 6: Dominant expert per class
        ax6 = axes[1, 2]

        dominant_expert = contribution_matrix.argmax(axis=1)  # [100]
        head_dominant = dominant_expert[:69]
        tail_dominant = dominant_expert[69:]

        head_counts = [np.sum(head_dominant == e) for e in range(3)]
        tail_counts = [np.sum(tail_dominant == e) for e in range(3)]

        bars1 = ax6.bar(
            x - width / 2,
            head_counts,
            width,
            label="Head (0-68)",
            alpha=0.8,
            edgecolor="black",
            color="skyblue",
        )
        bars2 = ax6.bar(
            x + width / 2,
            tail_counts,
            width,
            label="Tail (69-99)",
            alpha=0.8,
            edgecolor="black",
            color="coral",
        )

        ax6.set_ylabel("Number of Classes", fontsize=11)
        ax6.set_title("Dominant Expert Count by Group", fontweight="bold", fontsize=12)
        ax6.set_xticks(x)
        ax6.set_xticklabels(expert_names)
        ax6.legend()
        ax6.grid(alpha=0.3, axis="y")

        # Add values
        for bars, vals in [(bars1, head_counts), (bars2, tail_counts)]:
            for bar, val in zip(bars, vals):
                ax6.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.5,
                    f"{int(val)}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.suptitle(
            "Softmax Distribution & Expert Contribution Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        plt.savefig(
            save_path / "softmax_contribution.png", dpi=150, bbox_inches="tight"
        )
        print(f"Saved softmax contribution analysis to {save_path}")
        plt.close()


def plot_mixture_smoothing_effect(
    posteriors: torch.Tensor,
    gating: GatingNetwork,
    labels: torch.Tensor,
    save_path: Path,
):
    """
    Chứng minh mixture làm "smooth" distribution hơn.

    Show: Mixture có entropy cao hơn → predictions không quá "peaky"
    """

    with torch.no_grad():
        print("Analyzing mixture smoothing effect...")

        weights, _ = gating(posteriors)
        mixture = gating.get_mixture_posterior(posteriors, weights)

        # Compute entropy per sample
        def entropy(probs):
            return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        # Average entropy per expert
        expert_entropies = []
        for e in range(posteriors.shape[1]):
            ent = entropy(posteriors[:, e])
            expert_entropies.append(ent)

        mixture_entropy = entropy(mixture)

        # Sample some classes
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))

        for plot_idx, class_id in enumerate([0, 1, 34, 35, 68, 69, 84, 99, 50]):
            ax = axes[plot_idx // 3, plot_idx % 3]

            # Find samples of this class
            mask = labels == class_id
            if mask.sum() == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"No samples for class {class_id}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"Class {class_id} (Empty)", fontweight="bold")
                continue

            # Get posteriors for this class
            class_posteriors = posteriors[mask].mean(dim=0).cpu().numpy()  # [E, C]
            class_mixture = mixture[mask].mean(dim=0).cpu().numpy()  # [C]

            # Get top-10 classes
            top10_indices = class_mixture.argsort()[-10:][::-1]
            top10_values = class_mixture[top10_indices]

            # Plot
            x = np.arange(len(top10_indices))
            colors = ["green" if c == class_id else "steelblue" for c in top10_indices]

            bars = ax.bar(x, top10_values, color=colors, alpha=0.7, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"C{c}" for c in top10_indices], rotation=45, fontsize=8
            )
            ax.set_ylabel("Probability", fontsize=9)
            ax.set_title(f"Class {class_id} (Mixture)", fontweight="bold", fontsize=10)
            ax.grid(alpha=0.3, axis="y")

            # Add entropy annotation
            ent_val = -(class_mixture * np.log(class_mixture + 1e-8)).sum()
            ax.text(
                0.95,
                0.95,
                f"H={ent_val:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
            )

        plt.suptitle(
            "Mixture Distribution per Class (Showing Top-10)",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path / "mixture_smoothing.png", dpi=150, bbox_inches="tight")
        print(f"Saved mixture smoothing analysis to {save_path}")
        plt.close()


def main():
    """Main function."""
    print("=" * 70)
    print("SOFTMAX DISTRIBUTION & CONTRIBUTION ANALYSIS")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    logits_dir = Path("./outputs/logits/cifar100_lt_if100")
    splits_dir = Path("./data/cifar100_lt_if100_splits_fixed")

    expert_names = ["CE", "LogitAdjust", "BalancedSoftmax"]
    expert_logit_names = ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"]

    # Load expert logits
    logits_list = []
    for expert_name in expert_logit_names:
        logits_path = logits_dir / expert_name / "test_logits.pt"
        logits_e = torch.load(logits_path, map_location="cpu").float()
        logits_list.append(logits_e)

    logits = torch.stack(logits_list, dim=0).transpose(0, 1)  # [N, E, C]
    posteriors = torch.softmax(logits, dim=-1)

    print(f"  Loaded: {logits.shape[0]} samples")

    # Load labels
    import torchvision

    dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=False)
    with open(splits_dir / "test_indices.json", "r") as f:
        indices = json.load(f)
    labels = torch.tensor([dataset.targets[i] for i in indices])
    print(f"[OK] Loaded {len(labels)} labels")

    # Load gating
    print("\n2. Loading gating model...")
    gating_path = Path("./checkpoints/gating_map/cifar100_lt_if100/final_gating.pth")
    checkpoint = torch.load(gating_path, map_location="cpu")

    gating = GatingNetwork(num_experts=3, num_classes=100, routing="dense")
    gating.load_state_dict(checkpoint["model_state_dict"])
    gating.eval()
    print("[OK] Gating loaded")

    # Create output directory
    output_dir = Path("./results/comprehensive_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analyses
    print("\n3. Comparing softmax distributions...")
    compare_softmax_distributions(posteriors, gating, labels, expert_names, output_dir)

    print("\n4. Analyzing mixture smoothing effect...")
    plot_mixture_smoothing_effect(posteriors, gating, labels, output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
