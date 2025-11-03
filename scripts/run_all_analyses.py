#!/usr/bin/env python3
"""
Run All Analyses Script
=======================

Runs all comprehensive analyses for the MoE-L2R paper:

1. Comprehensive MoE analysis (variance, MI, calibration, oracle)
2. RC curve coverage analysis (per-group)
3. Summary table generation

Usage:
    python scripts/run_all_analyses.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run all analyses in sequence."""
    print("="*70)
    print("Running All MoE-L2R Analyses")
    print("="*70)
    
    try:
        # 1. Comprehensive MoE Analysis
        print("\n" + "="*70)
        print("Step 1: Comprehensive MoE Analysis")
        print("="*70)
        from scripts.moe_l2r_comprehensive_analysis import main as analyze_moe
        analyze_moe()
        
        # 2. Detailed Analysis (A-G)
        print("\n" + "="*70)
        print("Step 2: Detailed MoE Analysis (A-G)")
        print("="*70)
        from scripts.moe_detailed_analysis import main as analyze_detailed
        analyze_detailed()
        
        # 3. Counterfactual Analysis (H-I)
        print("\n" + "="*70)
        print("Step 3: Counterfactual & Ablation Analysis (H-I)")
        print("="*70)
        from scripts.moe_counterfactual_analysis import main as analyze_counterfactual
        analyze_counterfactual()
        
        # 4. RC Coverage Analysis
        print("\n" + "="*70)
        print("Step 4: RC Curve Coverage Analysis")
        print("="*70)
        from scripts.rc_coverage_analysis import analyze_rc_coverage
        analyze_rc_coverage()
        
        # 5. Generate Summary Tables
        print("\n" + "="*70)
        print("Step 5: Generate Summary Tables")
        print("="*70)
        from scripts.generate_summary_tables import main as generate_tables
        generate_tables()
        
        print("\n" + "="*70)
        print("All Analyses Complete!")
        print("="*70)
        print("\nGenerated files:")
        print("  - Comprehensive analysis: results/moe_analysis/cifar100_lt_if100/")
        print("  - Detailed plots (A-G): results/moe_analysis/cifar100_lt_if100/")
        print("  - Counterfactual plots (H-I): results/moe_analysis/cifar100_lt_if100/")
        print("  - Summary tables: results/moe_analysis/cifar100_lt_if100/table_*.csv")
        print("  - LaTeX tables: results/moe_analysis/cifar100_lt_if100/summary_tables.tex")
        print("="*70)
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

