"""
Calibration & Ablation Analysis
==============================

Chứng minh:
1. Calibration benefits từ ensemble
2. Ablation studies (single expert vs ensemble, routing strategies)
3. Impact của từng component

Usage:
    python -m src.visualize.calibration_ablation
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
from sklearn.metrics import brier_score_loss, cohen_kappa_score
import torchvision

from src.models.gating_network_map import GatingNetwork
from src.metrics.calibration import calculate_ece


def compute_calibration_metrics(
    posteriors: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15
) -> Dict[str, float]:
    """
    Compute ECE, Brier score, và accuracy.
    """
    # Get predictions và confidences
    predictions = posteriors.argmax(dim=-1)  # [N]
    confidences = posteriors.max(dim=-1)[0]  # [N]
    correct = (predictions == labels).float()  # [N]
    
    # ECE
    ece = calculate_ece(posteriors, labels, num_bins)
    
    # Brier score
    brier = brier_score_loss(labels.cpu().numpy(), posteriors.cpu().numpy())
    
    # Accuracy
    acc = correct.mean().item()
    
    return {
        'ece': ece,
        'brier': brier,
        'accuracy': acc
    }


def analyze_calibration(
    posteriors: torch.Tensor,
    labels: torch.Tensor,
    gating: GatingNetwork,
    expert_names: List[str],
    save_path: Path
):
    """
    So sánh calibration của từng expert vs mixture.
    """
    print("Analyzing calibration...")
    
    with torch.no_grad():
        num_experts = posteriors.shape[1]
        
        # Compute metrics cho từng expert
        expert_metrics = []
        for e in range(num_experts):
            metrics = compute_calibration_metrics(posteriors[:, e], labels)
            expert_metrics.append({
                'name': expert_names[e],
                **metrics
            })
        
        # Compute metrics cho mixture
        weights, _ = gating(posteriors)
        mixture = gating.get_mixture_posterior(posteriors, weights)
        mixture_metrics = compute_calibration_metrics(mixture, labels)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: ECE comparison
        ax1 = axes[0, 0]
        methods = expert_names + ['Mixture']
        eces = [m['ece'] for m in expert_metrics] + [mixture_metrics['ece']]
        colors = ['steelblue'] * num_experts + ['green']
        
        bars = ax1.bar(methods, eces, color=colors, alpha=0.7, 
                       edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('ECE (Lower is Better)', fontsize=11)
        ax1.set_title('Expected Calibration Error', fontweight='bold', fontsize=12)
        ax1.grid(alpha=0.3, axis='y')
        
        for bar, ece in zip(bars, eces):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{ece:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Brier Score comparison
        ax2 = axes[0, 1]
        briers = [m['brier'] for m in expert_metrics] + [mixture_metrics['brier']]
        
        bars = ax2.bar(methods, briers, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Brier Score (Lower is Better)', fontsize=11)
        ax2.set_title('Brier Score', fontweight='bold', fontsize=12)
        ax2.grid(alpha=0.3, axis='y')
        
        for bar, brier in zip(bars, briers):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{brier:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 3: Reliability diagrams (top 3 classes)
        ax3 = axes[1, 0]
        
        # Compute reliability cho mixture
        confidences_mix = mixture.max(dim=-1)[0]
        correct_mix = (mixture.argmax(dim=-1) == labels).float()
        
        # Bin confidence
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences_mix.cpu().numpy(), bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_accs = []
        bin_confs = []
        for b in range(n_bins):
            mask = (bin_indices == b)
            if mask.sum() > 0:
                bin_accs.append(correct_mix[mask].mean().item())
                bin_confs.append(confidences_mix[mask].mean().item())
            else:
                bin_accs.append(0)
                bin_confs.append(0)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        ax3.plot(bin_centers, bin_accs, 'o-', label='Accuracy', linewidth=2,
                markersize=6, color='blue')
        ax3.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1)
        ax3.set_xlabel('Confidence', fontsize=11)
        ax3.set_ylabel('Accuracy', fontsize=11)
        ax3.set_title('Reliability Diagram (Mixture)', fontweight='bold', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        
        # Plot 4: Accuracy vs Confidence trade-off
        ax4 = axes[1, 1]
        
        # Sort by confidence
        confidences_sorted, indices = confidences_mix.sort(descending=True)
        correct_sorted = correct_mix[indices]
        
        # Cumulative accuracy
        cumulative_acc = correct_sorted.cumsum(0) / torch.arange(1, len(correct_sorted) + 1, 
                                                                device=correct_sorted.device)
        
        # Sample for plotting
        step = max(1, len(cumulative_acc) // 1000)
        x_plot = torch.arange(0, len(cumulative_acc), step).cpu()
        y_plot = cumulative_acc[::step].cpu()
        
        ax4.plot(x_plot, y_plot, linewidth=2, label='Mixture', color='green')
        ax4.set_xlabel('Top-K Predictions (by Confidence)', fontsize=11)
        ax4.set_ylabel('Cumulative Accuracy', fontsize=11)
        ax4.set_title('Accuracy vs Confidence Trade-off', fontweight='bold', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        plt.suptitle('Calibration Analysis: Single Experts vs Mixture', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration analysis to {save_path}")
        plt.close()


def ablation_study(
    posteriors: torch.Tensor,
    labels: torch.Tensor,
    gating: GatingNetwork,
    expert_names: List[str],
    save_path: Path
):
    """
    Ablation study: Impact của từng component.
    """
    print("Running ablation study...")
    
    with torch.no_grad():
        # 1. Single experts
        expert_accs = []
        for e in range(posteriors.shape[1]):
            pred = posteriors[:, e].argmax(dim=-1)
            acc = (pred == labels).float().mean().item()
            expert_accs.append(acc)
        
        # 2. Mixture (với gating)
        weights, _ = gating(posteriors)
        mixture = gating.get_mixture_posterior(posteriors, weights)
        mixture_acc = (mixture.argmax(dim=-1) == labels).float().mean().item()
        
        # 3. Uniform mixture (baseline - no gating)
        uniform_weights = torch.ones_like(weights) / weights.shape[1]
        uniform_mixture = (uniform_weights.unsqueeze(-1) * posteriors).sum(dim=1)
        uniform_acc = (uniform_mixture.argmax(dim=-1) == labels).float().mean().item()
        
        # 4. Best single expert
        best_single = max(expert_accs)
        best_single_idx = expert_accs.index(best_single)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        methods = expert_names + ['Uniform\nMixture', 'Gated\nMixture']
        accuracies = expert_accs + [uniform_acc, mixture_acc]
        colors = ['steelblue'] * len(expert_names) + ['orange', 'green']
        
        bars = ax1.bar(methods, accuracies, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.set_title('Ablation: Accuracy Comparison', fontweight='bold', fontsize=12)
        ax1.grid(alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.0])
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Improvement analysis
        improvement_uniform = mixture_acc - uniform_acc
        improvement_best = mixture_acc - best_single
        
        ax1.text(0.5, 0.95, f'vs Uniform: +{improvement_uniform:.3f} | vs Best: +{improvement_best:.3f}',
                transform=ax1.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # Plot 2: Per-class improvement (top 20)
        ax2 = axes[0, 1]
        
        best_expert_pred = posteriors[:, best_single_idx].argmax(dim=-1)
        best_expert_correct = (best_expert_pred == labels).float()
        mixture_correct = (mixture.argmax(dim=-1) == labels).float()
        
        class_improvements = []
        for c in range(20):
            mask = (labels == c)
            if mask.sum() > 0:
                expert_acc_c = best_expert_correct[mask].mean().item()
                mixture_acc_c = mixture_correct[mask].mean().item()
                class_improvements.append(mixture_acc_c - expert_acc_c)
            else:
                class_improvements.append(0)
        
        colors_imp = ['green' if x > 0 else 'red' for x in class_improvements]
        ax2.bar(range(20), class_improvements, color=colors_imp, alpha=0.7,
               edgecolor='black')
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Class Index (First 20)', fontsize=11)
        ax2.set_ylabel('Accuracy Improvement', fontsize=11)
        ax2.set_title('Per-Class Improvement (vs Best Expert)', fontweight='bold', fontsize=12)
        ax2.grid(alpha=0.3, axis='y')
        
        # Plot 3: Head vs Tail Performance
        ax3 = axes[1, 0]
        
        head_mask = labels < 69
        tail_mask = labels >= 69
        
        # Best single expert
        head_acc_best = best_expert_correct[head_mask].mean().item()
        tail_acc_best = best_expert_correct[tail_mask].mean().item()
        
        # Mixture
        head_acc_mix = mixture_correct[head_mask].mean().item()
        tail_acc_mix = mixture_correct[tail_mask].mean().item()
        
        x = np.arange(2)
        width = 0.35
        
        ax3.bar(x - width/2, [head_acc_best, tail_acc_best], width,
               label='Best Single Expert', color='steelblue', alpha=0.7)
        ax3.bar(x + width/2, [head_acc_mix, tail_acc_mix], width,
               label='Mixture', color='green', alpha=0.7)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Head (0-68)', 'Tail (69-99)'])
        ax3.set_ylabel('Accuracy', fontsize=11)
        ax3.set_title('Head vs Tail Performance', fontweight='bold', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3, axis='y')
        
        # Add values
        for i, (best_acc, mix_acc) in enumerate([(head_acc_best, head_acc_mix), 
                                                  (tail_acc_best, tail_acc_mix)]):
            ax3.text(i - width/2, best_acc + 0.01, f'{best_acc:.2f}', 
                    ha='center', va='bottom', fontsize=9)
            ax3.text(i + width/2, mix_acc + 0.01, f'{mix_acc:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Component Importance
        ax4 = axes[1, 1]
        
        components = ['Best\nExpert', 'Uniform\nMixture', 'Gating\n(Router)', 'Mixture\n(Final)']
        
        # Improvement từng step
        improvements = [
            best_single,  # Baseline
            uniform_acc - best_single,  # Uniform mixture helps?
            mixture_acc - uniform_acc,  # Gating helps?
            mixture_acc  # Final
        ]
        
        colors_cmp = ['grey', 'orange', 'green', 'darkgreen']
        bars = ax4.bar(components, improvements, color=colors_cmp, alpha=0.7,
                      edgecolor='black', linewidth=1.5)
        
        ax4.set_ylabel('Cumulative Accuracy', fontsize=11)
        ax4.set_title('Component Contribution', fontweight='bold', fontsize=12)
        ax4.grid(alpha=0.3, axis='y')
        
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Ablation Study: Component Impact', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ablation study to {save_path}")
        plt.close()


def main():
    print("="*70)
    print("CALIBRATION & ABLATION ANALYSIS")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    logits_dir = Path('./outputs/logits/cifar100_lt_if100')
    splits_dir = Path('./data/cifar100_lt_if100_splits_fixed')
    
    expert_names = ['CE', 'LogitAdjust', 'BalancedSoftmax']
    
    # Load test logits
    expert_logit_names = ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']
    logits_list = []
    for expert_name in expert_logit_names:
        logits_path = logits_dir / expert_name / 'test_logits.pt'
        logits_e = torch.load(logits_path, map_location='cpu').float()
        logits_list.append(logits_e)
    
    logits = torch.stack(logits_list, dim=0).transpose(0, 1)
    posteriors = torch.softmax(logits, dim=-1)
    
    # Load labels
    dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    with open(splits_dir / 'test_indices.json', 'r') as f:
        indices = json.load(f)
    labels = torch.tensor([dataset.targets[i] for i in indices])

    print(f"[OK] Loaded {len(labels)} test samples")

    # Load gating
    print("\n2. Loading gating model...")
    gating_path = Path('./checkpoints/gating_map/cifar100_lt_if100/final_gating.pth')
    checkpoint = torch.load(gating_path, map_location='cpu')
    
    gating = GatingNetwork(num_experts=3, num_classes=100, routing='dense')
    gating.load_state_dict(checkpoint['model_state_dict'])
    gating.eval()
    print("[OK] Gating loaded")
    
    # Create output directory
    output_dir = Path('./results/comprehensive_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analyses
    print("\n3. Analyzing calibration...")
    analyze_calibration(posteriors, labels, gating, expert_names,
                       output_dir / 'calibration_analysis.png')
    
    print("\n4. Running ablation study...")
    ablation_study(posteriors, labels, gating, expert_names,
                  output_dir / 'ablation_study.png')
    
    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETED!")
    print(f"Results saved to: {output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()

