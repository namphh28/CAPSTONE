"""
Detailed Expert Contribution Analysis
=====================================

Tập trung vào CONTRIBUTION của từng expert:
1. Softmax heatmaps per expert vs mixture
2. Contribution weights per prediction
3. Mixture smoothing effect (entropy increase)
4. Per-class expert dominance

Usage:
    python -m src.visualize.detailed_contribution
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple
import json

from src.models.gating_network_map import GatingNetwork


def plot_expert_contribution_heatmaps(
    posteriors: torch.Tensor,
    gating: GatingNetwork,
    labels: torch.Tensor,
    expert_names: List[str],
    save_path: Path,
):
    """Plot detailed contribution heatmaps showing expert impact."""

    with torch.no_grad():
        print("Creating expert contribution heatmaps...")

        weights, _ = gating(posteriors)
        mixture = gating.get_mixture_posterior(posteriors, weights)

        # Select representative samples
        np.random.seed(42)
        sample_idx = np.random.choice(len(labels), 5, replace=False)

        fig = plt.figure(figsize=(20, 16))

        for plot_idx, idx in enumerate(sample_idx):
            ax = fig.add_subplot(3, 5, plot_idx + 1)

            true_class = labels[idx].item()
            true_group = "Head" if true_class < 69 else "Tail"

            # Get expert posteriors for this sample
            expert_probs = posteriors[idx].cpu().numpy()  # [E, C]
            mixture_prob = mixture[idx].cpu().numpy()  # [C]
            sample_weights = weights[idx].cpu().numpy()  # [E]

            # Top-10 classes
            top10_classes = mixture_prob.argsort()[-10:][::-1]

            # Create heatmap
            heatmap_data = np.zeros((4, 10))  # [expert_index, class]
            heatmap_data[:3] = expert_probs[:, top10_classes]
            heatmap_data[3] = mixture_prob[top10_classes]

            # Labels
            row_labels = expert_names + ["Mixture"]
            col_labels = [f"C{c}" for c in top10_classes]

            sns.heatmap(
                heatmap_data,
                ax=ax,
                cmap="YlGnBu",
                vmin=0,
                vmax=1,
                xticklabels=col_labels,
                yticklabels=row_labels,
                cbar=False,
                annot=False,
                fmt=".2f",
            )
            ax.set_title(
                f"Idx {idx}: True=C{true_class} ({true_group})",
                fontweight="bold",
                fontsize=10,
            )
            ax.set_xlabel("Class", fontsize=9)
            ax.set_ylabel("Expert", fontsize=9)

        # Plot weight distributions
        for plot_idx, idx in enumerate(sample_idx):
            ax = fig.add_subplot(3, 5, plot_idx + 6)

            true_class = labels[idx].item()
            sample_weights = weights[idx].cpu().numpy()

            bars = ax.bar(
                expert_names,
                sample_weights,
                color=["steelblue", "orange", "green"],
                alpha=0.7,
                edgecolor="black",
            )
            ax.set_ylabel("Weight", fontsize=9)
            ax.set_title(f"Sample {idx} Weights", fontweight="bold", fontsize=10)
            ax.set_ylim([0, 1.0])
            ax.grid(alpha=0.3, axis="y")

            # Add values
            for bar, val in zip(bars, sample_weights):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Plot entropy comparison
        def entropy(probs):
            return -(probs * torch.log(probs + 1e-8)).sum()

        entropies_per_sample = []
        for plot_idx, idx in enumerate(sample_idx):
            ax = fig.add_subplot(3, 5, plot_idx + 11)

            # Compute entropy for each expert
            expert_ents = []
            for e in range(3):
                ent = entropy(posteriors[idx, e]).item()
                expert_ents.append(ent)

            mix_ent = entropy(mixture[idx]).item()

            # Plot
            x = np.arange(len(expert_names) + 1)
            values = expert_ents + [mix_ent]
            labels_list = expert_names + ["Mixture"]

            bars = ax.bar(
                x,
                values,
                color=["steelblue", "orange", "green", "red"],
                alpha=0.7,
                edgecolor="black",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels_list, rotation=15)
            ax.set_ylabel("Entropy (bits)", fontsize=9)
            ax.set_title(f"Sample {idx} Entropy", fontweight="bold", fontsize=10)
            ax.grid(alpha=0.3, axis="y")

            # Add values
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.05,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.suptitle(
            "Expert Contribution: Per-Sample Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        plt.savefig(
            save_path / "expert_contribution_detail.png", dpi=150, bbox_inches="tight"
        )
        print(f"Saved detailed contribution to {save_path}")
        plt.close()


def plot_mixture_smoothing_comparison(
    posteriors: torch.Tensor,
    gating: GatingNetwork,
    labels: torch.Tensor,
    expert_names: List[str],
    save_path: Path,
):
    """Chứng minh Mixture smooth hơn Single experts."""

    with torch.no_grad():
        print("Comparing mixture smoothing effect...")

        weights, _ = gating(posteriors)
        mixture = gating.get_mixture_posterior(posteriors, weights)

        # Compute entropy
        def entropy(probs):
            return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        expert_entropies = [entropy(posteriors[:, e]) for e in range(3)]
        mixture_entropy = entropy(mixture)

        # Head vs Tail analysis
        head_mask = labels < 69
        tail_mask = labels >= 69

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Overall entropy comparison
        ax1 = axes[0, 0]

        expert_means = [torch.mean(ent).item() for ent in expert_entropies]
        mix_mean = torch.mean(mixture_entropy).item()

        methods = expert_names + ["Mixture"]
        values = expert_means + [mix_mean]
        colors = ["steelblue", "orange", "green", "red"]

        bars = ax1.bar(
            methods, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
        )
        ax1.set_ylabel("Mean Entropy (bits)", fontsize=12)
        ax1.set_title(
            "Overall Entropy: Experts vs Mixture", fontweight="bold", fontsize=13
        )
        ax1.grid(alpha=0.3, axis="y")

        # Add values and improvement
        for bar, val in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        # Improvement annotation
        best_expert_ent = max(expert_means)
        improvement = mix_mean - best_expert_ent
        ax1.text(
            0.5,
            0.95,
            f"Mixture higher by +{improvement:.3f} bits",
            transform=ax1.transAxes,
            ha="center",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
            fontsize=11,
        )

        # Plot 2: Head vs Tail entropy
        ax2 = axes[0, 1]

        head_expert_ents = [
            torch.mean(ent[head_mask]).item() for ent in expert_entropies
        ]
        tail_expert_ents = [
            torch.mean(ent[tail_mask]).item() for ent in expert_entropies
        ]
        head_mix_ent = torch.mean(mixture_entropy[head_mask]).item()
        tail_mix_ent = torch.mean(mixture_entropy[tail_mask]).item()

        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax2.bar(
            x - width / 2,
            head_expert_ents + [head_mix_ent],
            width,
            label="Head (0-68)",
            alpha=0.8,
            edgecolor="black",
        )
        bars2 = ax2.bar(
            x + width / 2,
            tail_expert_ents + [tail_mix_ent],
            width,
            label="Tail (69-99)",
            alpha=0.8,
            edgecolor="black",
        )

        ax2.set_ylabel("Mean Entropy", fontsize=12)
        ax2.set_title("Entropy: Head vs Tail", fontweight="bold", fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=15, ha="right")
        ax2.legend()
        ax2.grid(alpha=0.3, axis="y")

        # Plot 3: Confidence comparison
        ax3 = axes[1, 0]

        expert_confidences = [posteriors[:, e].max(dim=-1)[0] for e in range(3)]
        mixture_confidence = mixture.max(dim=-1)[0]

        conf_means = [torch.mean(conf).item() for conf in expert_confidences]
        mix_conf_mean = torch.mean(mixture_confidence).item()

        bars = ax3.bar(
            methods,
            conf_means + [mix_conf_mean],
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax3.set_ylabel("Mean Confidence", fontsize=12)
        ax3.set_title("Confidence: Experts vs Mixture", fontweight="bold", fontsize=13)
        ax3.grid(alpha=0.3, axis="y")

        for bar, val in zip(bars, conf_means + [mix_conf_mean]):
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        # Plot 4: Contribution strength
        ax4 = axes[1, 1]

        # Average weight magnitude
        weight_means = weights.mean(dim=0).cpu().numpy()
        weight_stds = weights.std(dim=0).cpu().numpy()

        x = np.arange(len(expert_names))
        bars = ax4.bar(
            x,
            weight_means,
            yerr=weight_stds,
            capsize=5,
            color=["steelblue", "orange", "green"],
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax4.set_ylabel("Mean Weight ± Std", fontsize=12)
        ax4.set_title("Expert Weight Distribution", fontweight="bold", fontsize=13)
        ax4.set_xticks(x)
        ax4.set_xticklabels(expert_names)
        ax4.axhline(1 / 3, color="red", linestyle="--", label="Perfect Balance")
        ax4.legend()
        ax4.grid(alpha=0.3, axis="y")

        # Add values
        for bar, mean_val, std_val in zip(bars, weight_means, weight_stds):
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + std_val + 0.02,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.suptitle(
            "Mixture Smoothing & Contribution Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        plt.savefig(
            save_path / "mixture_smoothing_comparison.png", dpi=150, bbox_inches="tight"
        )
        print(f"Saved smoothing comparison to {save_path}")
        plt.close()


def main():
    """Main function."""
    print("=" * 70)
    print("DETAILED EXPERT CONTRIBUTION ANALYSIS")
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

    logits = torch.stack(logits_list, dim=0).transpose(0, 1)
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
    print("\n3. Plotting expert contribution heatmaps...")
    plot_expert_contribution_heatmaps(
        posteriors, gating, labels, expert_names, output_dir
    )

    print("\n4. Comparing mixture smoothing...")
    plot_mixture_smoothing_comparison(
        posteriors, gating, labels, expert_names, output_dir
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
