"""
Visualization for MAP Plugin Results
=====================================

Plot RC curves, group comparisons, và debug analysis.

Usage:
    python src/visualize/plot_map_results.py
    python src/visualize/plot_map_results.py --compare balanced worst
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import List, Dict

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_results(results_dir: Path, objective: str = None) -> Dict:
    """Load results from JSON files."""
    # RC curve
    if objective:
        rc_path = results_dir / objective / 'rc_curve.json'
        eval_path = results_dir / objective / 'evaluation_test.json'
        params_path = Path(str(results_dir).replace('results', 'checkpoints')) / objective / 'map_parameters.json'
    else:
        # Files in root
        rc_path = results_dir / 'rc_curve.json'
        eval_path = results_dir / 'evaluation_test.json'
        params_path = Path(str(results_dir).replace('results', 'checkpoints')) / 'map_parameters.json'
    
    with open(rc_path, 'r') as f:
        rc_data = json.load(f)
    
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    return {
        'rc': rc_data,
        'eval': eval_data,
        'params': params,
        'objective': objective or params.get('objective', 'unknown')
    }


def plot_single_rc_curve(
    results: Dict,
    objective: str,
    save_path: Path,
    xlim: tuple = (0, 1.0)
):
    """Plot RC curve for single objective."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    rc_data = results['rc']
    rejection_rates = np.array(rc_data['rejection_rates'])
    selective_errors = np.array(rc_data['selective_errors'])
    aurc = rc_data['aurc']
    
    # Full range
    ax1.plot(rejection_rates, selective_errors, 'b-', linewidth=2.5, label=f'{objective.title()} (AURC={aurc:.4f})')
    ax1.fill_between(rejection_rates, 0, selective_errors, alpha=0.2)
    ax1.set_xlabel('Rejection Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Selective Error', fontsize=12, fontweight='bold')
    ax1.set_title(f'Error vs Rejection Rate (0-1)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    
    # Practical range (0.0-0.8)
    mask = rejection_rates <= xlim[1]
    ax2.plot(rejection_rates[mask], selective_errors[mask], 'b-', linewidth=2.5, label=f'{objective.title()}')
    ax2.set_xlabel('Rejection Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Selective Error', fontsize=12, fontweight='bold')
    ax2.set_title(f'Error vs Rejection Rate (0-{xlim[1]})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, xlim[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_group_rc_curves(
    results: Dict,
    objective: str,
    save_path: Path,
    xlim: tuple = (0, 0.8)
):
    """Plot group-wise RC curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    rc_data = results['rc']
    rejection_rates = np.array(rc_data['rejection_rates'])
    selective_errors = np.array(rc_data['selective_errors'])
    group_errors = rc_data['group_errors']
    
    # Filter to xlim
    mask = rejection_rates <= xlim[1]
    rej_filtered = rejection_rates[mask]
    
    # Plot groups
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    group_names = ['Head (0-49)', 'Tail (50-99)']
    
    for g_id in range(len(group_errors[0])):
        errors = np.array([ge[g_id] for ge in group_errors])[mask]
        ax.plot(rej_filtered, errors, 
                color=colors[g_id], linewidth=2.5, 
                label=f'Group {g_id}: {group_names[g_id]}',
                marker='o', markersize=3, markevery=5)
    
    # Overall
    ax.plot(rej_filtered, selective_errors[mask], 
            'k--', linewidth=2.5, label='Overall', alpha=0.7)
    
    ax.set_xlabel('Rejection Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Selective Error', fontsize=12, fontweight='bold')
    ax.set_title(f'Group-wise RC Curves ({objective.title()})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, xlim[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_comparison(
    results_list: List[Dict],
    objectives: List[str],
    save_path: Path,
    xlim: tuple = (0, 0.8)
):
    """Compare multiple objectives side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'balanced': '#2E86AB', 'worst': '#F18F01'}
    
    # Plot 1: Full range RC curves
    for results, obj in zip(results_list, objectives):
        rc_data = results['rc']
        rej = np.array(rc_data['rejection_rates'])
        err = np.array(rc_data['selective_errors'])
        aurc = rc_data['aurc']
        
        axes[0].plot(rej, err, color=colors.get(obj, 'gray'), 
                    linewidth=2.5, label=f'{obj.title()} (AURC={aurc:.4f})')
    
    axes[0].set_xlabel('Proportion of Rejections', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Error', fontsize=12, fontweight='bold')
    axes[0].set_title('Error vs Rejection Rate (0-1)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1.0)
    axes[0].set_ylim(0, 1.0)
    
    # Plot 2: Practical range
    for results, obj in zip(results_list, objectives):
        rc_data = results['rc']
        rej = np.array(rc_data['rejection_rates'])
        err = np.array(rc_data['selective_errors'])
        
        mask = rej <= xlim[1]
        axes[1].plot(rej[mask], err[mask], color=colors.get(obj, 'gray'),
                    linewidth=2.5, label=f'{obj.title()}')
    
    axes[1].set_xlabel('Proportion of Rejections', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Error', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Error vs Rejection Rate (0-{xlim[1]})', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, xlim[1])
    
    # Plot 3: AURC comparison
    aurc_full = []
    aurc_practical = []
    
    for results, obj in zip(results_list, objectives):
        rc_data = results['rc']
        rej = np.array(rc_data['rejection_rates'])
        err = np.array(rc_data['selective_errors'])
        
        # Full AURC
        aurc_full.append(rc_data['aurc'])
        
        # Practical AURC (0-xlim[1])
        mask = rej <= xlim[1]
        if mask.sum() > 1:
            aurc_prac = np.trapz(err[mask], rej[mask]) / xlim[1]
        else:
            aurc_prac = 0.0
        aurc_practical.append(aurc_prac)
    
    x_pos = np.arange(len(objectives))
    width = 0.35
    
    bars1 = axes[2].bar(x_pos - width/2, aurc_full, width, 
                       label='Full (0-1)', color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = axes[2].bar(x_pos + width/2, aurc_practical, width,
                       label=f'Practical (0.2-1.0)', color='#2E86AB', alpha=0.5, 
                       edgecolor='black', hatch='///')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add comparison text
    if len(objectives) == 2:
        diff_pct = ((aurc_full[1] - aurc_full[0]) / aurc_full[0]) * 100
        axes[2].text(0.5, max(aurc_full) * 0.9, 
                    f'Worst full AURC {diff_pct:+.1f}% vs Balanced',
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    axes[2].set_xlabel('', fontsize=12)
    axes[2].set_ylabel('AURC', fontsize=12, fontweight='bold')
    axes[2].set_title('AURC Comparison (Full vs 0.2-1.0)', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([obj.title() for obj in objectives])
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_accuracy_breakdown(
    results_list: List[Dict],
    objectives: List[str],
    save_path: Path
):
    """Plot accuracy breakdown by group."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract metrics
    data = []
    for results, obj in zip(results_list, objectives):
        eval_data = results['eval']
        overall_acc = eval_data['classification_metrics']['accuracy']
        head_acc = eval_data['group_metrics']['0']['accuracy']
        tail_acc = eval_data['group_metrics']['1']['accuracy']
        
        data.append({
            'objective': obj,
            'overall': overall_acc,
            'head': head_acc,
            'tail': tail_acc,
            'gap': head_acc - tail_acc
        })
    
    # Plot 1: Grouped bar chart
    x_pos = np.arange(len(objectives))
    width = 0.25
    
    axes[0].bar(x_pos - width, [d['overall'] for d in data], width, 
               label='Overall', color='#2E86AB', edgecolor='black')
    axes[0].bar(x_pos, [d['head'] for d in data], width,
               label='Head (0-49)', color='#06A77D', edgecolor='black')
    axes[0].bar(x_pos + width, [d['tail'] for d in data], width,
               label='Tail (50-99)', color='#A23B72', edgecolor='black')
    
    axes[0].set_xlabel('Objective', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy by Group', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([obj.title() for obj in objectives])
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 1.0)
    
    # Plot 2: Head-Tail gap
    bars = axes[1].bar([obj.title() for obj in objectives], [d['gap'] for d in data],
                      color='#F18F01', edgecolor='black', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    axes[1].set_xlabel('Objective', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Head - Tail Gap', fontsize=12, fontweight='bold')
    axes[1].set_title('Accuracy Gap (Head - Tail)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()


def print_debug_info(results: Dict, objective: str):
    """Print debug information."""
    print(f"\n{'='*70}")
    print(f"DEBUG: {objective.upper()}")
    print(f"{'='*70}")
    
    # Parameters
    params = results['params']
    print(f"\nParameters:")
    print(f"  λ = {params['lambda']:.3f}")
    print(f"  γ = {params['gamma']:.3f}")
    print(f"  ν = {params['nu']:.3f}")
    print(f"  α range: [{min(params['alpha']):.4f}, {max(params['alpha']):.4f}]")
    print(f"  μ range: [{min(params['mu']):.4f}, {max(params['mu']):.4f}]")
    
    # Classification metrics
    eval_data = results['eval']
    cm = eval_data['classification_metrics']
    print(f"\nClassification (without rejection):")
    print(f"  Accuracy: {cm['accuracy']:.4f}")
    print(f"  Top-5 Acc: {cm['top5_accuracy']:.4f}")
    print(f"  NLL: {cm['nll']:.4f}")
    print(f"  ECE: {cm['ece']:.4f}")
    
    # Group-wise
    print(f"\nGroup-wise:")
    for g_id, gm in eval_data['group_metrics'].items():
        print(f"  Group {g_id}: Acc={gm['accuracy']:.4f}, Count={gm['count']}, Weight={gm['weight_sum']:.4f}")
    
    # RC curve stats
    rc_data = results['rc']
    rej = np.array(rc_data['rejection_rates'])
    err = np.array(rc_data['selective_errors'])
    
    print(f"\nRC Curve:")
    print(f"  AURC (full): {rc_data['aurc']:.4f}")
    print(f"  Min error: {err.min():.4f}")
    print(f"  Max error: {err.max():.4f}")
    print(f"  Error at rej=0.0: {err[np.argmin(np.abs(rej - 0.0))]:.4f}")
    print(f"  Error at rej=0.2: {err[np.argmin(np.abs(rej - 0.2))]:.4f}")
    print(f"  Error at rej=0.5: {err[np.argmin(np.abs(rej - 0.5))]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, 
                       default='./results/map_plugin/cifar100_lt_if100',
                       help='Results directory')
    parser.add_argument('--compare', nargs='+', 
                       default=['balanced', 'worst'],
                       help='Objectives to compare')
    parser.add_argument('--xlim', type=float, default=0.8,
                       help='X-axis limit for practical range plots')
    parser.add_argument('--output_dir', type=str,
                       default='./outputs/visualizations/map_plugin',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("📊 MAP PLUGIN VISUALIZATION")
    print("="*70)
    
    # Load results
    print(f"\nLoading results from: {results_dir}")
    results_list = []
    
    # Try loading with objective subfolders first
    for obj in args.compare:
        try:
            results = load_results(results_dir, obj)
            results_list.append(results)
            print(f"  ✓ Loaded: {obj}")
        except Exception as e:
            print(f"  ✗ Failed to load {obj} from subfolder: {e}")
    
    # If no results, try loading from root (single result)
    if len(results_list) == 0:
        try:
            results = load_results(results_dir, objective=None)
            results_list.append(results)
            args.compare = [results['objective']]
            print(f"  ✓ Loaded from root: {results['objective']}")
        except Exception as e:
            print(f"  ✗ Failed to load from root: {e}")
    
    if len(results_list) == 0:
        print("\n❌ No results loaded!")
        return
    
    # Individual plots
    print(f"\n{'='*70}")
    print("Generating individual plots...")
    print(f"{'='*70}")
    
    for results, obj in zip(results_list, args.compare):
        print(f"\n{obj.upper()}:")
        
        # Single RC curve
        plot_single_rc_curve(
            results, obj,
            save_path=output_dir / f'rc_curve_{obj}.png',
            xlim=(0, args.xlim)
        )
        
        # Group-wise RC curves
        plot_group_rc_curves(
            results, obj,
            save_path=output_dir / f'rc_curve_groups_{obj}.png',
            xlim=(0, args.xlim)
        )
        
        # Debug info
        print_debug_info(results, obj)
    
    # Comparison plots
    if len(results_list) >= 2:
        print(f"\n{'='*70}")
        print("Generating comparison plots...")
        print(f"{'='*70}\n")
        
        # RC curve comparison
        plot_comparison(
            results_list, args.compare,
            save_path=output_dir / 'comparison_rc_curves.png',
            xlim=(0, args.xlim)
        )
        
        # Accuracy breakdown
        plot_accuracy_breakdown(
            results_list, args.compare,
            save_path=output_dir / 'comparison_accuracy.png'
        )
    
    print(f"\n{'='*70}")
    print("✅ VISUALIZATION COMPLETED!")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print("\n🎉 Done!")


if __name__ == '__main__':
    main()
