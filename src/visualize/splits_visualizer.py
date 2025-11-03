#!/usr/bin/env python3
"""
Visualize data distribution for all splits (train, expert, gating, test, val, tunev).
Shows per-class sample counts to verify long-tail properties are preserved.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional
import seaborn as sns


class SplitsDistributionVisualizer:
    """Visualize distribution of all dataset splits."""
    
    def __init__(self, splits_dir: str = "data/cifar100_lt_if100_splits_fixed"):
        self.splits_dir = Path(splits_dir)
        self.output_dir = Path("outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette for different splits
        self.colors = {
            'train': '#1f77b4',        # Blue
            'expert': '#ff7f0e',       # Orange
            'gating': '#2ca02c',       # Green
            'test': '#d62728',         # Red
            'val': '#9467bd',          # Purple
            'tunev': '#8c564b',        # Brown
        }
        
        # Line styles
        self.linestyles = {
            'train': '-',
            'expert': '-',
            'gating': '--',
            'test': '-.',
            'val': '-.',
            'tunev': ':'
        }
        
    def load_split_data(self, split_name: str, is_from_train: bool = True) -> Optional[Dict[int, int]]:
        """
        Load class distribution for a specific split.
        
        Args:
            split_name: Name of split (train, expert, gating, test, val, tunev)
            is_from_train: Whether indices come from CIFAR-100 train set
            
        Returns:
            Dictionary mapping class_id -> count
        """
        indices_file = self.splits_dir / f"{split_name}_indices.json"
        
        if not indices_file.exists():
            print(f"⚠️  {split_name} not found: {indices_file}")
            return None
        
        # Load indices
        with open(indices_file, 'r') as f:
            indices = json.load(f)
        
        # Load CIFAR-100 to get labels
        import torchvision
        if is_from_train:
            cifar_dataset = torchvision.datasets.CIFAR100(
                root="data", train=True, download=False
            )
        else:
            cifar_dataset = torchvision.datasets.CIFAR100(
                root="data", train=False, download=False
            )
        
        # Count classes
        class_counts = Counter()
        for idx in indices:
            if idx < len(cifar_dataset):
                _, label = cifar_dataset[idx]
                class_counts[label] += 1
        
        # Convert to sorted dict (class 0-99)
        distribution = {cls: class_counts.get(cls, 0) for cls in range(100)}
        
        print(f"✅ Loaded {split_name}: {len(indices):,} samples, "
              f"Head={distribution[0]}, Tail={distribution[99]}, "
              f"IF={distribution[0]/max(distribution[99], 1):.1f}")
        
        return distribution
    
    def visualize_all_splits(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of all data splits.
        Shows per-class distribution for train, expert, gating, test, val, tunev.
        """
        print("\n" + "="*80)
        print("VISUALIZING ALL DATA SPLITS")
        print("="*80 + "\n")
        
        # Load all splits
        splits_data = {}
        
        # Training splits (from CIFAR-100 train)
        for split_name in ['train', 'expert', 'gating']:
            data = self.load_split_data(split_name, is_from_train=True)
            if data is not None:
                splits_data[split_name] = data
        
        # Test splits (from CIFAR-100 test)
        for split_name in ['test', 'val', 'tunev']:
            data = self.load_split_data(split_name, is_from_train=False)
            if data is not None:
                splits_data[split_name] = data
        
        if not splits_data:
            print("❌ No splits found to visualize!")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('CIFAR-100-LT Data Distribution Across All Splits', 
                     fontsize=16, fontweight='bold')
        
        # 1. All splits on one plot (log scale)
        ax1 = axes[0, 0]
        for split_name, distribution in splits_data.items():
            classes = list(distribution.keys())
            counts = list(distribution.values())
            ax1.plot(classes, counts, 
                    label=f"{split_name.upper()} ({sum(counts):,})",
                    color=self.colors.get(split_name, 'gray'),
                    linestyle=self.linestyles.get(split_name, '-'),
                    linewidth=2,
                    marker='o' if split_name in ['expert', 'gating'] else None,
                    markersize=3,
                    alpha=0.8)
        
        ax1.set_xlabel('Class Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Samples (log scale)', fontsize=12, fontweight='bold')
        ax1.set_title('All Splits - Per-Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        
        # 2. Training splits comparison (linear scale)
        ax2 = axes[0, 1]
        train_splits = {k: v for k, v in splits_data.items() if k in ['train', 'expert', 'gating']}
        for split_name, distribution in train_splits.items():
            classes = list(distribution.keys())
            counts = list(distribution.values())
            ax2.plot(classes, counts,
                    label=f"{split_name.upper()} ({sum(counts):,})",
                    color=self.colors.get(split_name, 'gray'),
                    linestyle=self.linestyles.get(split_name, '-'),
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    alpha=0.8)
        
        ax2.set_xlabel('Class Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax2.set_title('Training Splits - Long-Tail Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        
        # 3. Test splits comparison (should be balanced)
        ax3 = axes[1, 0]
        test_splits = {k: v for k, v in splits_data.items() if k in ['test', 'val', 'tunev']}
        for split_name, distribution in test_splits.items():
            classes = list(distribution.keys())
            counts = list(distribution.values())
            ax3.plot(classes, counts,
                    label=f"{split_name.upper()} ({sum(counts):,})",
                    color=self.colors.get(split_name, 'gray'),
                    linestyle=self.linestyles.get(split_name, '-'),
                    linewidth=2,
                    marker='s',
                    markersize=4,
                    alpha=0.8)
        
        ax3.set_xlabel('Class Index', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax3.set_title('Test Splits - Balanced Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.set_ylim(bottom=0)
        
        # 4. Statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create statistics table
        stats_data = []
        for split_name, distribution in splits_data.items():
            total = sum(distribution.values())
            head_count = distribution[0]
            tail_count = distribution[99]
            imb_factor = head_count / max(tail_count, 1)
            min_count = min(distribution.values())
            max_count = max(distribution.values())
            
            stats_data.append([
                split_name.upper(),
                f"{total:,}",
                f"{head_count}",
                f"{tail_count}",
                f"{imb_factor:.1f}",
                f"{min_count}-{max_count}"
            ])
        
        table = ax4.table(
            cellText=stats_data,
            colLabels=['Split', 'Total', 'Head (C0)', 'Tail (C99)', 'IF', 'Range'],
            cellLoc='center',
            loc='center',
            colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        for i in range(1, len(stats_data) + 1):
            for j in range(6):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        ax4.set_title('Distribution Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "splits_distribution_complete.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {save_path}")
        
        plt.close()
        
        return save_path
    
    def visualize_expert_gating_comparison(self, save_path: Optional[str] = None):
        """
        Focused comparison of Expert vs Gating splits.
        Verify they maintain the same imbalance ratio.
        """
        print("\n" + "="*80)
        print("EXPERT vs GATING COMPARISON")
        print("="*80 + "\n")
        
        expert_dist = self.load_split_data('expert', is_from_train=True)
        gating_dist = self.load_split_data('gating', is_from_train=True)
        
        if expert_dist is None or gating_dist is None:
            print("❌ Expert or Gating split not found!")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Expert vs Gating Training Splits Comparison', 
                     fontsize=16, fontweight='bold')
        
        # 1. Absolute counts
        ax1 = axes[0]
        classes = list(range(100))
        expert_counts = [expert_dist[c] for c in classes]
        gating_counts = [gating_dist[c] for c in classes]
        
        ax1.plot(classes, expert_counts, 
                label=f'Expert ({sum(expert_counts):,} samples)',
                color=self.colors['expert'],
                linewidth=2.5,
                marker='o',
                markersize=5,
                alpha=0.8)
        ax1.plot(classes, gating_counts,
                label=f'Gating ({sum(gating_counts):,} samples)',
                color=self.colors['gating'],
                linewidth=2.5,
                marker='s',
                markersize=5,
                alpha=0.8)
        
        ax1.set_xlabel('Class Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax1.set_title('Absolute Sample Counts', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=11)
        
        # 2. Normalized distribution (verify same shape)
        ax2 = axes[1]
        expert_norm = np.array(expert_counts) / sum(expert_counts)
        gating_norm = np.array(gating_counts) / sum(gating_counts)
        
        ax2.plot(classes, expert_norm,
                label='Expert (normalized)',
                color=self.colors['expert'],
                linewidth=2.5,
                marker='o',
                markersize=5,
                alpha=0.8)
        ax2.plot(classes, gating_norm,
                label='Gating (normalized)',
                color=self.colors['gating'],
                linewidth=2.5,
                marker='s',
                markersize=5,
                alpha=0.8)
        
        ax2.set_xlabel('Class Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Proportion of Samples', fontsize=12, fontweight='bold')
        ax2.set_title('Normalized Distribution (Should Overlap)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=11)
        
        # Add statistics
        expert_if = expert_counts[0] / max(expert_counts[99], 1)
        gating_if = gating_counts[0] / max(gating_counts[99], 1)
        
        textstr = f'Expert IF: {expert_if:.1f}\nGating IF: {gating_if:.1f}\nRatio preserved: {"✅" if abs(expert_if - gating_if) < 1 else "⚠️"}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = self.output_dir / "expert_vs_gating_comparison.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison saved to: {save_path}")
        
        plt.close()
        
        return save_path
    
    def print_summary_statistics(self):
        """Print detailed summary statistics for all splits."""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80 + "\n")
        
        # Load all splits
        all_splits = {}
        for split_name in ['train', 'expert', 'gating']:
            data = self.load_split_data(split_name, is_from_train=True)
            if data is not None:
                all_splits[split_name] = ('train', data)
        
        for split_name in ['test', 'val', 'tunev']:
            data = self.load_split_data(split_name, is_from_train=False)
            if data is not None:
                all_splits[split_name] = ('test', data)
        
        for split_name, (source, distribution) in all_splits.items():
            total = sum(distribution.values())
            head = distribution[0]
            tail = distribution[99]
            imb_factor = head / max(tail, 1)
            min_count = min(distribution.values())
            max_count = max(distribution.values())
            unique_counts = len(set(distribution.values()))
            
            print(f"{split_name.upper():>10} (from CIFAR-100 {source}):")
            print(f"  Total samples:     {total:>6,}")
            print(f"  Head class (0):    {head:>6}")
            print(f"  Tail class (99):   {tail:>6}")
            print(f"  Imbalance factor:  {imb_factor:>6.1f}")
            print(f"  Range:             {min_count:>6} - {max_count}")
            print(f"  Unique counts:     {unique_counts:>6}")
            print()


def visualize_splits(splits_dir: str = "data/cifar100_lt_if100_splits_fixed", 
                     output_dir: str = "outputs/visualizations"):
    """
    Main function to visualize all dataset splits.
    
    Args:
        splits_dir: Directory containing split JSON files
        output_dir: Directory to save visualizations
    """
    visualizer = SplitsDistributionVisualizer(splits_dir)
    
    # Print statistics
    visualizer.print_summary_statistics()
    
    # Create visualizations
    visualizer.visualize_all_splits()
    visualizer.visualize_expert_gating_comparison()
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nCheck outputs in: {output_dir}/")
    print("  - splits_distribution_complete.png")
    print("  - expert_vs_gating_comparison.png")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize CIFAR-100-LT splits distribution")
    parser.add_argument('--splits-dir', type=str, 
                       default='data/cifar100_lt_if100_splits_fixed',
                       help='Directory containing split files')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    visualize_splits(args.splits_dir, args.output_dir)
