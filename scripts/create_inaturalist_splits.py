#!/usr/bin/env python3
"""
Quick script to generate iNaturalist 2018 splits from JSON files.
Usage: python scripts/create_inaturalist_splits.py --train-json path/to/train.json --val-json path/to/val.json
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.inaturalist2018_splits import create_inaturalist2018_splits


def main():
    parser = argparse.ArgumentParser(
        description="Generate iNaturalist 2018 dataset splits from JSON files"
    )
    parser.add_argument(
        '--train-json',
        type=str,
        required=True,
        help='Path to train.json (COCO format)'
    )
    parser.add_argument(
        '--val-json',
        type=str,
        required=True,
        help='Path to val.json (COCO format)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/inaturalist2018',
        help='Directory containing images (default: data/inaturalist2018)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/inaturalist2018_splits',
        help='Output directory for splits (default: data/inaturalist2018_splits)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--expert-ratio',
        type=float,
        default=0.9,
        help='Expert split ratio (default: 0.9)'
    )
    parser.add_argument(
        '--visualize-n',
        type=int,
        default=100,
        help='Number of last classes to visualize (default: 100)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file. If provided, all output will be saved to this file (default: None)'
    )
    
    args = parser.parse_args()
    
    print("Generating iNaturalist 2018 splits...")
    
    splits, class_weights = create_inaturalist2018_splits(
        train_json_path=args.train_json,
        val_json_path=args.val_json,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        expert_ratio=args.expert_ratio,
        visualize_last_n=args.visualize_n,
        log_file=args.log_file
    )
    
    print("\n✓ All splits generated successfully!")


if __name__ == "__main__":
    main()

