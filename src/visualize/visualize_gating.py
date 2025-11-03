"""
Visualization Script for Gating Network
========================================

Generate plots để hiểu behavior của gating:
1. Expert weights distribution
2. Uncertainty vs Error correlation
3. Load balancing over time
4. Feature importance
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from src.models.gating_network_map import (
    GatingNetwork,
    compute_uncertainty_for_map
)


def visualize_expert_weights(posteriors, gating, save_path='gating_weights.png'):
    """
    Visualize distribution of expert weights.
    
    Args:
        posteriors: [N, E, C]
        gating: GatingNetwork model
        save_path: where to save plot
    """
    weights, _ = gating(posteriors)  # [N, E]
    weights_np = weights.detach().cpu().numpy()
    
    E = weights.shape[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Histogram per expert
    for e in range(E):
        axes[0].hist(weights_np[:, e], bins=50, alpha=0.6, label=f'Expert {e+1}')
    axes[0].set_xlabel('Weight')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Expert Weight Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Heatmap (sample x expert)
    # Show first 100 samples
    sns.heatmap(weights_np[:100], ax=axes[1], cmap='viridis', 
                cbar_kws={'label': 'Weight'})
    axes[1].set_xlabel('Expert')
    axes[1].set_ylabel('Sample')
    axes[1].set_title('Expert Weights (First 100 Samples)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved expert weights plot to {save_path}")
    plt.close()


def visualize_uncertainty_vs_error(posteriors, gating, labels, save_path='uncertainty_vs_error.png'):
    """
    Visualize correlation giữa uncertainty và error.
    
    Expected: High uncertainty → High error rate
    """
    # Get predictions
    weights, _ = gating(posteriors)
    mixture = gating.get_mixture_posterior(posteriors, weights)
    predictions = mixture.argmax(dim=-1)
    
    # Compute uncertainty
    U = compute_uncertainty_for_map(posteriors, weights, mixture)
    
    # Compute correctness
    correct = (predictions == labels).float()
    
    # Convert to numpy
    U_np = U.detach().cpu().numpy()
    correct_np = correct.detach().cpu().numpy()
    
    # Binned analysis
    n_bins = 10
    bins = np.linspace(U_np.min(), U_np.max(), n_bins + 1)
    bin_indices = np.digitize(U_np, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_errors = []
    bin_centers = []
    bin_counts = []
    
    for b in range(n_bins):
        mask = (bin_indices == b)
        if mask.sum() > 0:
            bin_errors.append(1 - correct_np[mask].mean())
            bin_centers.append((bins[b] + bins[b+1]) / 2)
            bin_counts.append(mask.sum())
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Scatter
    axes[0].scatter(U_np, 1-correct_np, alpha=0.3, s=10)
    axes[0].set_xlabel('Uncertainty U(x)')
    axes[0].set_ylabel('Error (0=correct, 1=wrong)')
    axes[0].set_title('Uncertainty vs Error (Scatter)')
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Binned error rate
    axes[1].plot(bin_centers, bin_errors, marker='o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Uncertainty U(x) (binned)')
    axes[1].set_ylabel('Error Rate')
    axes[1].set_title('Error Rate vs Uncertainty (Binned)')
    axes[1].grid(alpha=0.3)
    
    # Add correlation
    corr = np.corrcoef(U_np, 1-correct_np)[0, 1]
    axes[1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=axes[1].transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved uncertainty vs error plot to {save_path}")
    print(f"   Correlation U vs Error: {corr:.4f}")
    plt.close()


def visualize_feature_importance(posteriors, gating, save_path='feature_importance.png'):
    """
    Analyze which features are most important.
    
    Method: Look at first layer weights magnitude
    """
    # Get first layer weights
    first_layer = gating.mlp.mlp[0]  # Linear layer
    weights = first_layer.weight.data.abs()  # [hidden, input_dim]
    
    # Average over hidden dimension
    importance = weights.mean(dim=0).cpu().numpy()  # [input_dim]
    
    # Get feature dimension info
    E = gating.num_experts
    C = gating.num_classes
    
    # Feature ranges
    posteriors_range = (0, E * C)
    expert_entropy_range = (E * C, E * C + E)
    expert_conf_range = (E * C + E, E * C + 2*E)
    expert_margin_range = (E * C + 2*E, E * C + 3*E)
    global_range = (E * C + 3*E, len(importance))
    
    # Aggregate importance per feature type
    feature_importance = {
        'Posteriors': importance[posteriors_range[0]:posteriors_range[1]].mean(),
        'Expert Entropy': importance[expert_entropy_range[0]:expert_entropy_range[1]].mean(),
        'Expert Confidence': importance[expert_conf_range[0]:expert_conf_range[1]].mean(),
        'Expert Margin': importance[expert_margin_range[0]:expert_margin_range[1]].mean(),
        'Global Features': importance[global_range[0]:global_range[1]].mean(),
    }
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    features = list(feature_importance.keys())
    values = list(feature_importance.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(features)))
    
    bars = ax.bar(features, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Average Weight Magnitude')
    ax.set_title('Feature Importance (First Layer Weights)')
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved feature importance plot to {save_path}")
    plt.close()


def visualize_expert_disagreement(posteriors, save_path='expert_disagreement.png'):
    """
    Visualize expert disagreement patterns.
    """
    B, E, C = posteriors.shape
    
    # Top-1 predictions per expert
    top1 = posteriors.argmax(dim=-1)  # [B, E]
    
    # Pairwise agreement matrix
    agreement = torch.zeros(E, E)
    for i in range(E):
        for j in range(E):
            agreement[i, j] = (top1[:, i] == top1[:, j]).float().mean()
    
    agreement_np = agreement.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Agreement heatmap
    sns.heatmap(agreement_np, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=axes[0],
                xticklabels=[f'E{i+1}' for i in range(E)],
                yticklabels=[f'E{i+1}' for i in range(E)])
    axes[0].set_title('Pairwise Expert Agreement')
    
    # Plot 2: Disagreement rate distribution
    disagreement_rates = []
    for b in range(B):
        unique_preds = torch.unique(top1[b]).numel()
        rate = (unique_preds - 1) / (E - 1)
        disagreement_rates.append(rate)
    
    axes[1].hist(disagreement_rates, bins=20, alpha=0.7, color='coral', edgecolor='black')
    axes[1].set_xlabel('Disagreement Rate')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Expert Disagreement Distribution')
    axes[1].axvline(np.mean(disagreement_rates), color='red', linestyle='--',
                   label=f'Mean: {np.mean(disagreement_rates):.3f}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved expert disagreement plot to {save_path}")
    print(f"   Mean disagreement rate: {np.mean(disagreement_rates):.4f}")
    plt.close()


def main():
    """Generate all visualizations with real expert logits."""
    print("="*70)
    print("📊 GATING NETWORK VISUALIZATION")
    print("="*70)
    
    # Check if logits exist
    logits_dir = Path('./outputs/logits/cifar100_lt_if100')
    if not logits_dir.exists():
        print("❌ Logits directory not found!")
        print("   Please train experts first: python3 src/train/train_expert.py")
        return
    
    # Load expert logits
    print("\n1. Loading expert logits...")
    expert_names = ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']
    split_name = 'val'
    
    logits_list = []
    for expert_name in expert_names:
        logits_path = logits_dir / expert_name / f"{split_name}_logits.pt"
        if not logits_path.exists():
            print(f"❌ {logits_path} not found!")
            return
        logits_e = torch.load(logits_path, map_location='cpu').float()
        logits_list.append(logits_e)
    
    logits = torch.stack(logits_list, dim=0).transpose(0, 1)  # [N, E, C]
    posteriors = torch.softmax(logits, dim=-1)
    
    N, E, C = posteriors.shape
    print(f"   ✓ Loaded: {N} samples, {E} experts, {C} classes")
    
    # Load labels
    print("\n2. Loading labels...")
    import torchvision
    dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    
    import json
    with open('./data/cifar100_lt_if100_splits_fixed/val_indices.json', 'r') as f:
        indices = json.load(f)
    
    labels = torch.tensor([dataset.targets[i] for i in indices])
    print(f"   ✓ Loaded {len(labels)} labels")
    
    # Initialize gating
    print("\n3. Initializing gating network...")
    gating = GatingNetwork(num_experts=E, num_classes=C)
    gating.eval()
    print("   ✓ Gating initialized (random weights)")
    
    # Create output directory
    vis_dir = Path('./outputs/visualizations/gating')
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n4. Generating visualizations → {vis_dir}/")
    
    # Generate plots
    print("\n   a) Expert weights distribution...")
    visualize_expert_weights(posteriors, gating, 
                            save_path=vis_dir / 'expert_weights.png')
    
    print("\n   b) Uncertainty vs Error correlation...")
    visualize_uncertainty_vs_error(posteriors, gating, labels,
                                  save_path=vis_dir / 'uncertainty_vs_error.png')
    
    print("\n   c) Feature importance...")
    visualize_feature_importance(posteriors, gating,
                                save_path=vis_dir / 'feature_importance.png')
    
    print("\n   d) Expert disagreement patterns...")
    visualize_expert_disagreement(posteriors,
                                 save_path=vis_dir / 'expert_disagreement.png')
    
    print("\n" + "="*70)
    print("✅ All visualizations generated!")
    print(f"📁 Check: {vis_dir}/")
    print("="*70)
    
    # Summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    weights, _ = gating(posteriors)
    mixture = gating.get_mixture_posterior(posteriors, weights)
    U = compute_uncertainty_for_map(posteriors, weights, mixture)
    predictions = mixture.argmax(dim=-1)
    
    print(f"   Mixture Accuracy: {(predictions == labels).float().mean():.4f}")
    print(f"   Mean Uncertainty: {U.mean():.4f} ± {U.std():.4f}")
    print(f"   Gating Entropy: {-(weights * torch.log(weights + 1e-8)).sum(dim=1).mean():.4f}")
    print(f"   Expert Usage: {weights.mean(dim=0)}")


if __name__ == '__main__':
    main()
