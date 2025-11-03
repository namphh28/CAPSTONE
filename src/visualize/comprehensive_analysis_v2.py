"""
Enhanced Comprehensive Analysis - V2
=====================================

Cải tiến from V1:
1. Full-class analysis (ALL 100 classes, not just 20)
2. Proper head/tail breakdown (0-68 vs 69-99)
3. Enhanced metrics and statistics
4. Better visualizations

Usage:
    python -m src.visualize.comprehensive_analysis_v2
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from src.models.gating_network_map import GatingNetwork


def analyze_full_class_performance(
    posteriors: torch.Tensor,
    gating: GatingNetwork,
    labels: torch.Tensor,
    expert_names: List[str],
    save_path: Path
):
    """Analyze performance across ALL 100 classes with proper head/tail breakdown."""
    
    with torch.no_grad():
        print("Analyzing full-class performance...")
        
        # Get mixture
        weights, _ = gating(posteriors)
        mixture = gating.get_mixture_posterior(posteriors, weights)
        mixture_pred = mixture.argmax(dim=-1)
        
        # Get expert predictions
        expert_preds = posteriors.argmax(dim=-1)  # [N, E]
        
        # Mask for head/tail
        head_mask = labels < 69
        tail_mask = labels >= 69
        
        # Calculate per-class accuracy for mixture
        mixture_class_accs = []
        head_accs = []
        tail_accs = []
        
        for c in range(100):
            mask = (labels == c)
            if mask.sum() > 0:
                acc = (mixture_pred[mask] == labels[mask]).float().mean().item()
                mixture_class_accs.append(acc)
                
                if c < 69:
                    head_accs.append(acc)
                else:
                    tail_accs.append(acc)
            else:
                mixture_class_accs.append(0.0)
        
        # Calculate expert accuracies
        expert_accs = []
        expert_head_accs = []
        expert_tail_accs = []
        
        for e in range(expert_preds.shape[1]):
            acc = (expert_preds[:, e] == labels).float().mean().item()
            expert_accs.append(acc)
            
            head_acc = (expert_preds[head_mask, e] == labels[head_mask]).float().mean().item()
            tail_acc = (expert_preds[tail_mask, e] == labels[tail_mask]).float().mean().item()
            
            expert_head_accs.append(head_acc)
            expert_tail_accs.append(tail_acc)
        
        # Mixture performance
        mixture_acc = (mixture_pred == labels).float().mean().item()
        mixture_head_acc = (mixture_pred[head_mask] == labels[head_mask]).float().mean().item()
        mixture_tail_acc = (mixture_pred[tail_mask] == labels[tail_mask]).float().mean().item()
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(24, 14))
        
        # ========================================================================
        # Plot 1: Per-Class Accuracy (ALL 100 classes)
        # ========================================================================
        ax1 = plt.subplot(3, 4, 1)
        
        colors = ['skyblue' if c < 69 else 'coral' for c in range(100)]
        bars = ax1.bar(range(100), mixture_class_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.axvline(68.5, color='red', linestyle='--', linewidth=2, label='Head/Tail Boundary')
        ax1.set_xlabel('Class Index', fontsize=11)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.set_title('Mixture Per-Class Accuracy (ALL 100 Classes)', fontweight='bold', fontsize=12)
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        
        # ========================================================================
        # Plot 2: Head vs Tail Distribution
        # ========================================================================
        ax2 = plt.subplot(3, 4, 2)
        
        bp = ax2.boxplot([head_accs, tail_accs])
        ax2.set_xticklabels(['Head (0-68)', 'Tail (69-99)'])
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_title('Accuracy Distribution: Head vs Tail', fontweight='bold', fontsize=12)
        ax2.grid(alpha=0.3, axis='y')
        
        # Add mean lines
        ax2.axhline(np.mean(head_accs), color='blue', linestyle=':', alpha=0.5)
        ax2.axhline(np.mean(tail_accs), color='red', linestyle=':', alpha=0.5)
        
        # Add statistics
        ax2.text(0.95, 0.95, f'Head Mean: {np.mean(head_accs):.3f}\nTail Mean: {np.mean(tail_accs):.3f}',
                transform=ax2.transAxes, va='top', ha='right', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # ========================================================================
        # Plot 3: Expert vs Mixture Comparison
        # ========================================================================
        ax3 = plt.subplot(3, 4, 3)
        
        methods = expert_names + ['Mixture']
        accuracies = expert_accs + [mixture_acc]
        
        colors_bars = ['steelblue'] * len(expert_names) + ['green']
        bars = ax3.bar(methods, accuracies, color=colors_bars, alpha=0.7, 
                      edgecolor='black', linewidth=1.5)
        
        ax3.set_ylabel('Overall Accuracy', fontsize=11)
        ax3.set_title('All Experts vs Mixture', fontweight='bold', fontsize=12)
        ax3.set_ylim([0, max(1.0, max(accuracies) * 1.1)])
        ax3.grid(alpha=0.3, axis='y')
        
        # Add values
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # ========================================================================
        # Plot 4: Head vs Tail Performance Comparison
        # ========================================================================
        ax4 = plt.subplot(3, 4, 4)
        
        head_perfs = expert_head_accs + [mixture_head_acc]
        tail_perfs = expert_tail_accs + [mixture_tail_acc]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, head_perfs, width, label='Head (0-68)', 
                       alpha=0.8, edgecolor='black')
        bars2 = ax4.bar(x + width/2, tail_perfs, width, label='Tail (69-99)',
                       alpha=0.8, edgecolor='black')
        
        ax4.set_ylabel('Accuracy', fontsize=11)
        ax4.set_title('Head vs Tail Performance', fontweight='bold', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods, rotation=15, ha='right')
        ax4.legend()
        ax4.grid(alpha=0.3, axis='y')
        
        # ========================================================================
        # Plot 5: Mixture Weights by Class Group
        # ========================================================================
        ax5 = plt.subplot(3, 4, 5)
        
        head_weights = weights[head_mask].mean(dim=0).cpu().numpy()
        tail_weights = weights[tail_mask].mean(dim=0).cpu().numpy()
        
        x = np.arange(len(expert_names))
        bars1 = ax5.bar(x - width/2, head_weights, width, label='Head classes',
                       alpha=0.8, edgecolor='black')
        bars2 = ax5.bar(x + width/2, tail_weights, width, label='Tail classes',
                       alpha=0.8, edgecolor='black')
        
        ax5.set_ylabel('Mean Gating Weight', fontsize=11)
        ax5.set_title('Expert Weights: Head vs Tail', fontweight='bold', fontsize=12)
        ax5.set_xticks(x)
        ax5.set_xticklabels(expert_names)
        ax5.legend()
        ax5.grid(alpha=0.3, axis='y')
        
        # Add values
        for bars, vals in [(bars1, head_weights), (bars2, tail_weights)]:
            for bar, val in zip(bars, vals):
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # ========================================================================
        # Plot 6: Error Rate Distribution
        # ========================================================================
        ax6 = plt.subplot(3, 4, 6)
        
        errors = [1 - acc for acc in mixture_class_accs]
        colors_err = ['skyblue' if c < 69 else 'coral' for c in range(100)]
        
        ax6.bar(range(100), errors, color=colors_err, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax6.axvline(68.5, color='red', linestyle='--', linewidth=2, label='Head/Tail Boundary')
        ax6.set_xlabel('Class Index', fontsize=11)
        ax6.set_ylabel('Error Rate', fontsize=11)
        ax6.set_title('Error Rate Distribution', fontweight='bold', fontsize=12)
        ax6.legend()
        ax6.grid(alpha=0.3, axis='y')
        
        # ========================================================================
        # Plot 7: Accuracy Heatmap (Selected classes)
        # ========================================================================
        ax7 = plt.subplot(3, 4, 7)
        
        # Sample representative classes
        sample_head = np.linspace(0, 68, 20, dtype=int).tolist()
        sample_tail = np.linspace(69, 99, 10, dtype=int).tolist()
        selected = sorted(sample_head + sample_tail)
        
        acc_matrix = np.array([mixture_class_accs[c] for c in selected])
        acc_matrix = acc_matrix.reshape(-1, 1)
        
        sns.heatmap(acc_matrix.T, ax=ax7, cmap='RdYlGn', vmin=0, vmax=1,
                   xticklabels=[str(c) for c in selected], yticklabels=['Mixture'],
                   cbar_kws={'label': 'Accuracy'})
        ax7.set_xlabel('Class Index (Sampled)', fontsize=11)
        ax7.set_title('Accuracy Heatmap (Representative)', fontweight='bold', fontsize=12)
        
        # ========================================================================
        # Plot 8: Expert Contribution by Class Group
        # ========================================================================
        ax8 = plt.subplot(3, 4, 8)
        
        # For each class group, which expert dominates?
        contribution = []
        for c in range(100):
            mask = (labels == c)
            if mask.sum() > 0:
                class_weights = weights[mask].mean(dim=0)  # [E]
                dominant = class_weights.argmax().item()
                contribution.append({
                    'class': c,
                    'dominant': dominant,
                    'weights': class_weights.cpu().numpy()
                })
        
        # Aggregate by head/tail
        head_dominant = [c['dominant'] for c in contribution if c['class'] < 69]
        tail_dominant = [c['dominant'] for c in contribution if c['class'] >= 69]
        
        # Count
        head_counts = [head_dominant.count(e) for e in range(3)]
        tail_counts = [tail_dominant.count(e) for e in range(3)]
        
        x = np.arange(3)
        ax8.bar(x - width/2, head_counts, width, label='Head classes', alpha=0.8)
        ax8.bar(x + width/2, tail_counts, width, label='Tail classes', alpha=0.8)
        ax8.set_xticks(x)
        ax8.set_xticklabels(expert_names)
        ax8.set_ylabel('Number of Classes', fontsize=11)
        ax8.set_title('Dominant Expert by Class Group', fontweight='bold', fontsize=12)
        ax8.legend()
        ax8.grid(alpha=0.3, axis='y')
        
        # ========================================================================
        # Plot 9-12: Summary Statistics
        # ========================================================================
        ax9 = plt.subplot(3, 4, 9)
        ax10 = plt.subplot(3, 4, 10)
        ax11 = plt.subplot(3, 4, 11)
        ax12 = plt.subplot(3, 4, 12)
        
        # Fill with statistics
        for ax in [ax9, ax10, ax11, ax12]:
            ax.axis('off')
        
        # Overall summary
        summary_text = f"""
        MIXTURE SUMMARY
        ===============
        
        Overall Performance:
        --------------------
        Accuracy: {mixture_acc:.3f}
        
        Head Classes (0-68):
          Accuracy: {mixture_head_acc:.3f}
          Mean: {np.mean(head_accs):.3f}
          Std: {np.std(head_accs):.3f}
          Min: {np.min(head_accs):.3f}
          Max: {np.max(head_accs):.3f}
        
        Tail Classes (69-99):
          Accuracy: {mixture_tail_acc:.3f}
          Mean: {np.mean(tail_accs):.3f}
          Std: {np.std(tail_accs):.3f}
          Min: {np.min(tail_accs):.3f}
          Max: {np.max(tail_accs):.3f}
        """
        
        ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        # Expert comparison
        best_single = max(expert_accs)
        improvement = mixture_acc - best_single
        
        expert_text = f"""
        BEST SINGLE vs MIXTURE
        ======================
        
        Best Single Expert:
        {expert_names[expert_accs.index(best_single)]}: {best_single:.3f}
        
        Mixture:
        {mixture_acc:.3f}
        
        Improvement:
        +{improvement:.3f}
        
        Relative: +{improvement/best_single*100:.1f}%
        """
        
        ax10.text(0.1, 0.5, expert_text, fontsize=10, family='monospace',
                  verticalalignment='center')
        
        # Weight analysis
        weight_text = f"""
        GATING WEIGHTS ANALYSIS
        =======================
        
        Mean Weights (Overall):
        """
        for name, mean_w in zip(expert_names, weights.mean(dim=0).cpu().numpy()):
            weight_text += f"\n  {name}: {mean_w:.3f}"
        
        weight_text += f"""
        
        Effective Experts:
        {np.exp(-(weights * torch.log(weights + 1e-8)).sum(dim=1).mean().item()):.2f}
        
        Load Balance (Ideal=0.33):
        {weights.mean(dim=0).std().item():.3f}
        """
        
        ax11.text(0.1, 0.5, weight_text, fontsize=10, family='monospace',
                 verticalalignment='center')
        
        # Class distribution
        labels_np = labels.cpu().numpy()
        head_count = np.sum(labels_np < 69)
        tail_count = np.sum(labels_np >= 69)
        
        dist_text = f"""
        CLASS DISTRIBUTION
        ==================
        
        Head Classes (0-68):
        {len(head_accs)} classes
        {head_count} samples
        
        Tail Classes (69-99):
        {len(tail_accs)} classes
        {tail_count} samples
        
        Total:
        100 classes
        {len(labels)} samples
        """
        
        ax12.text(0.1, 0.5, dist_text, fontsize=10, family='monospace',
                 verticalalignment='center')
        
        plt.suptitle('Full-Class Performance Analysis (All 100 Classes)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved full-class analysis to {save_path}")
        plt.close()


def main():
    """Main function."""
    print("="*70)
    print("ENHANCED COMPREHENSIVE ANALYSIS V2")
    print("="*70)
    
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
    
    print(f"  Loaded: {logits.shape[0]} samples, {logits.shape[1]} experts")
    
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
    print("[OK] Gating model loaded")
    
    # Create output directory
    output_dir = Path("./results/comprehensive_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run enhanced analysis
    print("\n3. Running full-class analysis...")
    analyze_full_class_performance(
        posteriors, gating, labels, expert_names,
        output_dir / "full_class_analysis.png"
    )
    
    print("\n" + "="*70)
    print("ENHANCED ANALYSIS COMPLETED!")
    print(f"Results saved to: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()

