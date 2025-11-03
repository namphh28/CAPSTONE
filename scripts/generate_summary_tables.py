#!/usr/bin/env python3
"""
Generate Summary Tables for Paper
==================================

Generates tables summarizing all analysis results for the paper.

Tables generated:
1. MoE Ablation: Expert vs Uniform vs Gating vs Oracle
2. Calibration Comparison: ECE/NLL/Brier per method
3. Gating Statistics: Entropy, effective experts, load balance
4. RC/AURC Summary: Balanced and worst-group metrics

Usage:
    python scripts/generate_summary_tables.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.moe_l2r_comprehensive_analysis import Config


CFG = Config()


def load_comprehensive_results() -> Dict:
    """Load comprehensive analysis results."""
    from scripts.moe_l2r_comprehensive_analysis import Config as AnalysisConfig
    results_path = Path(AnalysisConfig().output_dir) / "comprehensive_analysis_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    return results


def load_rc_results() -> Dict:
    """Load RC curve results."""
    from scripts.moe_l2r_comprehensive_analysis import Config as AnalysisConfig
    results_path = Path(AnalysisConfig().results_dir) / "ltr_plugin_gating_balanced.json"
    if not results_path.exists():
        return {}
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    return results


def generate_moe_ablation_table(results: Dict) -> pd.DataFrame:
    """Generate MoE ablation table: Expert vs Uniform vs Gating vs Oracle."""
    oracle_data = results.get('oracle_comparison', {})
    
    methods = ['oracle', 'uniform', 'gating']
    method_labels = ['Oracle@E', 'Uniform-Mix', 'Gating-Mix']
    
    data = {
        'Method': [],
        'Accuracy': [],
        'ECE': [],
        'NLL': [],
        'Brier': [],
        'Head Acc': [],
        'Tail Acc': [],
    }
    
    for method, label in zip(methods, method_labels):
        if method in oracle_data:
            data['Method'].append(label)
            data['Accuracy'].append(oracle_data[method]['acc'])
            data['ECE'].append(oracle_data[method]['ece'])
            data['NLL'].append(oracle_data[method]['nll'])
            data['Brier'].append(oracle_data[method]['brier'])
            
            group_accs = oracle_data[method]['group_accs']
            if len(group_accs) >= 2:
                data['Head Acc'].append(group_accs[0])
                data['Tail Acc'].append(group_accs[1])
            else:
                data['Head Acc'].append(None)
                data['Tail Acc'].append(None)
    
    # Add individual experts
    calibration = results.get('calibration', {})
    for expert_name in CFG.expert_names:
        if expert_name in calibration:
            data['Method'].append(expert_name)
            cal = calibration[expert_name]
            # For individual experts, we need to get acc from somewhere else
            # For now, use placeholder
            data['Accuracy'].append(None)  # Would need to load from actual predictions
            data['ECE'].append(cal['ece'])
            data['NLL'].append(cal['nll'])
            data['Brier'].append(cal['brier'])
            data['Head Acc'].append(None)
            data['Tail Acc'].append(None)
    
    df = pd.DataFrame(data)
    return df


def generate_calibration_table(results: Dict) -> pd.DataFrame:
    """Generate calibration comparison table."""
    calibration = results.get('calibration', {})
    
    data = {
        'Method': [],
        'ECE': [],
        'NLL': [],
        'Brier': [],
        'Head ECE': [],
        'Tail ECE': [],
    }
    
    for method, cal in calibration.items():
        data['Method'].append(method)
        data['ECE'].append(cal['ece'])
        data['NLL'].append(cal['nll'])
        data['Brier'].append(cal['brier'])
        
        group_metrics = cal.get('group_metrics', {})
        if 'group_0' in group_metrics:
            data['Head ECE'].append(group_metrics['group_0']['ece'])
        else:
            data['Head ECE'].append(None)
        
        if 'group_1' in group_metrics:
            data['Tail ECE'].append(group_metrics['group_1']['ece'])
        else:
            data['Tail ECE'].append(None)
    
    df = pd.DataFrame(data)
    return df


def generate_gating_statistics_table(results: Dict) -> pd.DataFrame:
    """Generate gating statistics table."""
    gating_stats = results.get('gating_statistics', {})
    
    data = {
        'Metric': [],
        'Value': [],
    }
    
    data['Metric'].append('Mean Weight Entropy')
    data['Value'].append(gating_stats.get('mean_entropy', None))
    
    data['Metric'].append('Mean Effective Experts')
    data['Value'].append(gating_stats.get('mean_effective_experts', None))
    
    data['Metric'].append('Load Balance Std')
    data['Value'].append(gating_stats.get('load_balance_std', None))
    
    # Per-expert mean weights
    mean_weights = gating_stats.get('mean_weights', [])
    for i, expert_name in enumerate(CFG.expert_names):
        if i < len(mean_weights):
            data['Metric'].append(f'Mean Weight ({expert_name})')
            data['Value'].append(mean_weights[i])
    
    # Group statistics
    group_stats = gating_stats.get('group_stats', {})
    for g in range(2):  # head and tail
        group_key = f'group_{g}'
        if group_key in group_stats:
            gs = group_stats[group_key]
            data['Metric'].append(f'Group {g} Mean Entropy')
            data['Value'].append(gs.get('mean_entropy', None))
            data['Metric'].append(f'Group {g} Mean Effective Experts')
            data['Value'].append(gs.get('mean_effective_experts', None))
    
    df = pd.DataFrame(data)
    return df


def generate_rc_summary_table(rc_results: Dict) -> pd.DataFrame:
    """Generate RC/AURC summary table."""
    if not rc_results or 'rc_curve' not in rc_results:
        return pd.DataFrame()
    
    rc_curve = rc_results['rc_curve']
    
    data = {
        'Split': [],
        'AURC (Balanced)': [],
        'AURC (Worst-group)': [],
        'AURC Balanced (ρ≤0.8)': [],
        'AURC Worst (ρ≤0.8)': [],
    }
    
    for split in ['val', 'test']:
        if split in rc_curve:
            data['Split'].append(split)
            split_data = rc_curve[split]
            data['AURC (Balanced)'].append(split_data.get('aurc_balanced', None))
            data['AURC (Worst-group)'].append(split_data.get('aurc_worst_group', None))
            data['AURC Balanced (ρ≤0.8)'].append(
                split_data.get('aurc_balanced_coverage_ge_0_2', None)
            )
            data['AURC Worst (ρ≤0.8)'].append(
                split_data.get('aurc_worst_group_coverage_ge_0_2', None)
            )
    
    df = pd.DataFrame(data)
    return df


def format_table_for_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Format DataFrame as LaTeX table."""
    if df.empty:
        return ""
    
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{tab:{label}}}\n"
    latex += "\\begin{tabular}{" + "c" * len(df.columns) + "}\n"
    latex += "\\toprule\n"
    
    # Header
    latex += " & ".join(df.columns) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Rows
    for _, row in df.iterrows():
        values = []
        for val in row:
            if val is None:
                values.append("---")
            elif isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        latex += " & ".join(values) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def main():
    """Generate all summary tables."""
    print("="*70)
    print("Generating Summary Tables")
    print("="*70)
    
    # Load results
    print("\n1. Loading results...")
    comprehensive_results = load_comprehensive_results()
    rc_results = load_rc_results()
    
    # Generate tables
    print("\n2. Generating tables...")
    
    tables = {}
    
    # Table 1: MoE Ablation
    print("   Generating MoE ablation table...")
    tables['moe_ablation'] = generate_moe_ablation_table(comprehensive_results)
    
    # Table 2: Calibration
    print("   Generating calibration table...")
    tables['calibration'] = generate_calibration_table(comprehensive_results)
    
    # Table 3: Gating Statistics
    print("   Generating gating statistics table...")
    tables['gating_stats'] = generate_gating_statistics_table(comprehensive_results)
    
    # Table 4: RC Summary
    print("   Generating RC summary table...")
    tables['rc_summary'] = generate_rc_summary_table(rc_results)
    
    # Save tables
    print("\n3. Saving tables...")
    from scripts.moe_l2r_comprehensive_analysis import Config as AnalysisConfig
    output_dir = Path(AnalysisConfig().output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    for table_name, df in tables.items():
        if not df.empty:
            csv_path = output_dir / f"table_{table_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"   Saved {csv_path}")
    
    # Save as LaTeX
    latex_path = output_dir / "summary_tables.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% Summary Tables for Paper\n")
        f.write("% Generated by scripts/generate_summary_tables.py\n\n")
        
        f.write(format_table_for_latex(
            tables['moe_ablation'],
            "MoE Ablation: Expert vs Uniform vs Gating vs Oracle",
            "moe_ablation"
        ))
        f.write("\n\n")
        
        f.write(format_table_for_latex(
            tables['calibration'],
            "Calibration Comparison: ECE/NLL/Brier",
            "calibration"
        ))
        f.write("\n\n")
        
        f.write(format_table_for_latex(
            tables['gating_stats'],
            "Gating Network Statistics",
            "gating_stats"
        ))
        f.write("\n\n")
        
        f.write(format_table_for_latex(
            tables['rc_summary'],
            "RC/AURC Summary",
            "rc_summary"
        ))
        f.write("\n")
    
    print(f"   Saved LaTeX tables to {latex_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("Tables Generated:")
    print("="*70)
    for table_name, df in tables.items():
        print(f"\n{table_name}:")
        if not df.empty:
            print(df.to_string())
        else:
            print("  (empty)")
    
    print("\n" + "="*70)
    print("Table Generation Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

