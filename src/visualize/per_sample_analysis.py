"""
Per-Sample Detailed Analysis
============================

Show 10 samples with:
1. Expert posteriors (all 100 classes) for each expert
2. Gating weights (how much each expert contributes)
3. Mixture distribution (after combining)
4. Prediction and decision (accept/reject)

Usage:
    python -m src.visualize.per_sample_analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import json

from src.models.gating_network_map import GatingNetwork


def plot_sample_analysis(
    posteriors: torch.Tensor,
    gating: GatingNetwork,
    labels: torch.Tensor,
    expert_names: List[str],
    save_path: Path,
    num_samples: int = 10,
):
    """Plot detailed analysis for each sample."""

    with torch.no_grad():
        print("Creating per-sample analysis...")

        # Select samples
        np.random.seed(42)
        sample_indices = np.random.choice(len(labels), num_samples, replace=False)

        # Get mixture
        weights, _ = gating(posteriors)
        mixture = gating.get_mixture_posterior(posteriors, weights)

        # Create figure with subplots
        fig = plt.figure(figsize=(24, 14))
        gs = fig.add_gridspec(num_samples, 4, hspace=0.4, wspace=0.3)

        for plot_idx, idx in enumerate(sample_indices):
            # Get data for this sample
            true_class = labels[idx].item()
            expert_probs = posteriors[idx].cpu().numpy()  # [E, C]
            mixture_prob = mixture[idx].cpu().numpy()  # [C]
            sample_weights = weights[idx].cpu().numpy()  # [E]

            # Predictions
            expert_preds = expert_probs.argmax(axis=1)
            mixture_pred = mixture_prob.argmax()

            # ========================================================================
            # Plot 1: Expert Posteriors (All 100 Classes)
            # ========================================================================
            ax1 = fig.add_subplot(gs[plot_idx, 0])

            # Plot expert posteriors
            for e_idx, (name, expert_prob) in enumerate(
                zip(expert_names, expert_probs)
            ):
                true_prob = expert_prob[true_class]
                ax1.plot(
                    expert_prob,
                    label=name,
                    linewidth=1.5,
                    alpha=0.7,
                    color=["steelblue", "orange", "green"][e_idx],
                )
                # Mark true class
                ax1.axvline(true_class, color="red", linestyle="--", alpha=0.5)
                ax1.plot(
                    [true_class], [true_prob], "ro", markersize=4, markeredgecolor="red"
                )

            ax1.set_xlim([0, 100])
            ax1.set_ylabel("Probability", fontsize=8)
            ax1.set_title(
                f"Sample {idx}: True={true_class} | \nPred: {mixture_pred}",
                fontweight="bold",
                fontsize=9,
            )
            ax1.grid(alpha=0.3, axis="y")
            ax1.set_xlabel("Class Index", fontsize=8)

            # Add legend only for first sample
            if plot_idx == 0:
                ax1.legend(loc="upper right", fontsize=7)

            # ========================================================================
            # Plot 2: Gating Weights
            # ========================================================================
            ax2 = fig.add_subplot(gs[plot_idx, 1])

            bars = ax2.bar(
                expert_names,
                sample_weights,
                color=["steelblue", "orange", "green"],
                alpha=0.7,
                edgecolor="black",
                linewidth=1.5,
            )
            ax2.set_ylabel("Weight", fontsize=9)
            ax2.set_title("Gating Weights", fontweight="bold", fontsize=9)
            ax2.set_ylim([0, 1.0])
            ax2.grid(alpha=0.3, axis="y")

            # Add values
            for bar, val in zip(bars, sample_weights):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            # ========================================================================
            # Plot 3: Mixture Distribution
            # ========================================================================
            ax3 = fig.add_subplot(gs[plot_idx, 2])

            ax3.plot(mixture_prob, color="purple", linewidth=1.5, alpha=0.7)
            ax3.axvline(true_class, color="red", linestyle="--", label="True")
            ax3.axvline(mixture_pred, color="black", linestyle=":", label="Pred")

            # Mark true and predicted
            true_prob = mixture_prob[true_class]
            pred_prob = mixture_prob[mixture_pred]
            ax3.plot([true_class], [true_prob], "ro", markersize=5)
            ax3.plot([mixture_pred], [pred_prob], "ks", markersize=5)

            ax3.set_xlim([0, 100])
            ax3.set_ylabel("Probability", fontsize=8)
            ax3.set_title(
                f"Mixture: True={true_class}({true_prob:.3f}) | Pred={mixture_pred}({pred_prob:.3f})",
                fontweight="bold",
                fontsize=9,
            )
            ax3.grid(alpha=0.3, axis="y")
            ax3.set_xlabel("Class Index", fontsize=8)

            # Add legend only for first sample
            if plot_idx == 0:
                ax3.legend(loc="upper right", fontsize=7)

            # ========================================================================
            # Plot 4: Summary Statistics
            # ========================================================================
            ax4 = fig.add_subplot(gs[plot_idx, 3])
            ax4.axis("off")

            # Prepare summary text
            true_group = "Head" if true_class < 69 else "Tail"
            pred_group = "Head" if mixture_pred < 69 else "Tail"
            is_correct = mixture_pred == true_class

            # Find dominant expert
            dominant_expert_idx = sample_weights.argmax()
            dominant_expert_name = expert_names[dominant_expert_idx]

            summary_text = f"""
            SAMPLE {idx}
            ================
            
            True Class: {true_class} ({true_group})
            Predicted: {mixture_pred} ({pred_group})
            Correct: {"YES" if is_correct else "NO"}
            
            Expert Predictions:
            - {expert_names[0]}: {expert_preds[0]} {"(T)" if expert_preds[0] == true_class else "(F)"}
            - {expert_names[1]}: {expert_preds[1]} {"(T)" if expert_preds[1] == true_class else "(F)"}
            - {expert_names[2]}: {expert_preds[2]} {"(T)" if expert_preds[2] == true_class else "(F)"}
            
            Gating Weights:
            - {expert_names[0]}: {sample_weights[0]:.3f}
            - {expert_names[1]}: {sample_weights[1]:.3f}
            - {expert_names[2]}: {sample_weights[2]:.3f}
            
            Dominant: {dominant_expert_name}
            
            True Prob: {true_prob:.4f}
            Top-3: {mixture_prob.argsort()[-3:][::-1].tolist()}
            """

            ax4.text(
                0.05,
                0.95,
                summary_text,
                transform=ax4.transAxes,
                fontsize=8,
                family="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.suptitle(
            f"Per-Sample Analysis: Expert Posteriors → Gating Weights → Mixture Distribution",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.savefig(save_path / "per_sample_analysis.png", dpi=150, bbox_inches="tight")
        print(f"Saved per-sample analysis to {save_path}")
        plt.close()


def main():
    """Main function."""
    print("=" * 70)
    print("PER-SAMPLE ANALYSIS")
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

    # Run analysis
    print("\n3. Creating per-sample analysis...")
    plot_sample_analysis(posteriors, gating, labels, expert_names, output_dir, num_samples=10)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

