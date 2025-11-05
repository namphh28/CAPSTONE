#!/usr/bin/env python3
"""
Expert Training Script for AR-GSE
Trains all expert models (CE, LogitAdjust, BalancedSoftmax) for the AR-GSE ensemble.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent / 'src'))

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train expert models for AR-GSE ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cifar100_lt_if100', 'inaturalist2018'],
        default='cifar100_lt_if100',
        help='Dataset to train on (default: cifar100_lt_if100)'
    )
    
    parser.add_argument(
        '--expert',
        type=str,
        choices=['ce', 'logitadjust', 'balsoftmax', 'all'],
        default='all',
        help='Expert type to train (or "all" for all experts)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of training epochs'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory for checkpoints'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from existing checkpoint if available'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without training'
    )
    
    parser.add_argument(
        '--use-expert-split',
        action='store_true',
        default=True,
        help='Use expert split (90%% of train) for training (default: True)'
    )
    
    parser.add_argument(
        '--use-full-train',
        action='store_true',
        help='Use full training set instead of expert split'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file. If provided, all output will be saved to this file'
    )
    
    return parser.parse_args()

def setup_training_environment(args):
    """Setup training environment and configurations."""
    try:
        from src.train.train_expert import CONFIG, DATASET_CONFIGS
        import torch
        
        # Set CONFIG based on dataset argument
        if hasattr(args, 'dataset') and args.dataset:
            dataset_config = DATASET_CONFIGS[args.dataset]
            CONFIG["dataset"].update(dataset_config)
            print(f"✓ Using dataset: {args.dataset}")
            print(f"  Classes: {dataset_config['num_classes']}")
            print(f"  Backbone: {dataset_config['backbone']}")
            print(f"  Batch size: {dataset_config['batch_size']}")
        
        # Setup device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
            
        print("🚀 AR-GSE Expert Training Pipeline")
        print(f"Device: {device}")
        print(f"Dataset: {CONFIG['dataset']['name']}")
        
        if args.verbose:
            print(f"PyTorch version: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU: {torch.cuda.get_device_name()}")
        
        return device
        
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        print("Please ensure you're running from the project root directory.")
        sys.exit(1)

def apply_overrides(expert_configs, args):
    """Apply command-line overrides to expert configurations."""
    if not any([args.epochs, args.lr, args.batch_size, args.output_dir]):
        return expert_configs
    
    overridden_configs = expert_configs.copy()
    
    for expert_key in overridden_configs:
        config = overridden_configs[expert_key].copy()
        
        if args.epochs:
            config['epochs'] = args.epochs
        if args.lr:
            config['lr'] = args.lr
        # Note: batch_size and output_dir would need to be implemented in the original train_expert.py
        
        overridden_configs[expert_key] = config
    
    return overridden_configs

def train_single_expert_wrapper(expert_key, args):
    """Wrapper for training a single expert with error handling."""
    try:
        from src.train.train_expert import train_single_expert
        
        # Determine which split to use
        use_expert_split = not args.use_full_train  # Default True unless --use-full-train
        
        if args.verbose:
            print(f"\n📋 Training configuration for {expert_key}:")
            from src.train.train_expert import EXPERT_CONFIGS
            config = EXPERT_CONFIGS[expert_key]
            for key, value in config.items():
                print(f"  {key}: {value}")
            print(f"  use_expert_split: {use_expert_split}")
        
        if args.dry_run:
            print(f"🔍 [DRY RUN] Would train expert: {expert_key}")
            print(f"    Using {'expert split (90% train)' if use_expert_split else 'full train'}")
            return f"checkpoints/experts/cifar100_lt_if100/{expert_key}_model.pth"
        
        model_path = train_single_expert(
            expert_key, 
            use_expert_split=use_expert_split,
            override_epochs=args.epochs,
            override_batch_size=args.batch_size
        )
        return model_path
        
    except Exception as e:
        raise Exception(f"Failed to train {expert_key}: {str(e)}")

def main():
    """Main function for expert training."""
    args = parse_arguments()
    
    # Setup logging if log_file is provided
    original_stdout = sys.stdout
    log_file_handle = None
    
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_handle = open(log_path, 'w', encoding='utf-8')
        
        # Create a class that writes to both stdout and log file
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            
            def flush(self):
                for f in self.files:
                    f.flush()
        
        sys.stdout = TeeOutput(original_stdout, log_file_handle)
        print(f"\n{'='*80}")
        print(f"LOGGING TO FILE: {log_path}")
        print(f"STARTED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
    
    try:
        print("=" * 60)
        print("AR-GSE EXPERT TRAINING")
        print("=" * 60)
        
        # Determine split usage
        use_expert_split = not args.use_full_train
        if use_expert_split:
            print("📊 Training Mode: Using EXPERT split (90% of train)")
            print("   - Trains on 9,719 samples (expert split)")
            print("   - Validates on 1,000 samples (balanced val)")
            print("   - Uses reweighted metrics for validation")
        else:
            print("📊 Training Mode: Using FULL train set")
            print("   - Trains on 10,847 samples (full train)")
            print("   - Validates on 1,000 samples (balanced val)")
            print("   - Uses reweighted metrics for validation")
        print()
        
        # Setup environment
        setup_training_environment(args)
        
        from src.train.train_expert import EXPERT_CONFIGS
        
        # Determine which experts to train
        if args.expert == 'all':
            experts_to_train = list(EXPERT_CONFIGS.keys())
        else:
            experts_to_train = [args.expert]
        
        print(f"Experts to train: {experts_to_train}")
        
        if args.dry_run:
            print("\n🔍 DRY RUN MODE - No actual training will be performed")
        
        # Train experts
        results = {}
        
        for expert_key in experts_to_train:
            print(f"\n{'='*40}")
            print(f"🎯 Training Expert: {expert_key.upper()}")
            print(f"{'='*40}")
            
            try:
                model_path = train_single_expert_wrapper(expert_key, args)
                results[expert_key] = {'status': 'success', 'path': model_path}
                print(f"✅ Successfully trained {expert_key}")
                if args.verbose:
                    print(f"   Model saved to: {model_path}")
                    
            except Exception as e:
                print(f"❌ Failed to train {expert_key}: {e}")
                results[expert_key] = {'status': 'failed', 'error': str(e)}
                
                if args.verbose:
                    import traceback
                    print("   Full error traceback:")
                    traceback.print_exc()
                continue
        
        # Print summary
        print(f"\n{'='*60}")
        print("🏁 TRAINING SUMMARY")
        print(f"{'='*60}")
        
        successful = 0
        for expert_key, result in results.items():
            status = "✅" if result['status'] == 'success' else "❌"
            print(f"{status} {expert_key}: {result['status']}")
            if result['status'] == 'success':
                successful += 1
                if args.verbose:
                    print(f"    Path: {result['path']}")
            else:
                print(f"    Error: {result['error']}")
        
        print(f"\nSuccessfully trained {successful}/{len(experts_to_train)} experts")
        
        if successful == len(experts_to_train):
            print("\n🎉 All experts trained successfully!")
            print("You can now proceed to the next step: gating model training")
        elif successful > 0:
            print(f"\n⚠️  Partial success: {successful} experts trained")
            print("You may need to retry failed experts or proceed with available ones")
        else:
            print("\n❌ No experts were trained successfully")
            print("Please check the errors above and resolve any issues")
            sys.exit(1)
            
        if args.log_file:
            print(f"\n{'='*80}")
            print(f"COMPLETED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"LOG FILE SAVED: {args.log_file}")
            print(f"{'='*80}")
        
    except ImportError as e:
        print(f"\n❌ Error importing training modules: {e}")
        print("\nPlease ensure:")
        print("1. You're running from the project root directory")
        print("2. All dependencies are installed (pip install -r requirements.txt)")
        print("3. The src/train/train_expert.py file exists")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Restore stdout and close log file
        if log_file_handle:
            sys.stdout = original_stdout
            log_file_handle.close()

if __name__ == "__main__":
    main()
