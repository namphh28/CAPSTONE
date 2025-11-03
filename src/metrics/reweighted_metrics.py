#!/usr/bin/env python3
"""
Reweighted metrics for long-tail evaluation.

This module implements metrics that are reweighted by the training distribution,
which is the CORRECT way to report results for long-tail datasets in papers.

Key Concept:
- Test set is BALANCED (equal samples per class)
- Training set is LONG-TAIL (imbalanced)
- When computing metrics, we REWEIGHT by train distribution
- This reflects real-world performance where classes appear with train frequency
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


class ReweightedMetrics:
    """
    Compute reweighted metrics for long-tail evaluation.
    
    Example Usage:
        metrics = ReweightedMetrics(class_weights)
        
        # After inference
        results = metrics.compute_metrics(
            predictions=preds,
            targets=labels,
            confidences=confs
        )
        
        print(f"Reweighted Accuracy: {results['reweighted_accuracy']:.2%}")
    """
    
    def __init__(self, class_weights: np.ndarray, num_classes: int = 100):
        """
        Initialize with class weights from training distribution.
        
        Args:
            class_weights: Array of shape (num_classes,) with weights summing to 1
            num_classes: Number of classes (default 100 for CIFAR-100)
        """
        self.class_weights = np.array(class_weights)
        self.num_classes = num_classes
        
        assert len(self.class_weights) == num_classes, \
            f"class_weights length {len(self.class_weights)} != num_classes {num_classes}"
        assert np.abs(self.class_weights.sum() - 1.0) < 1e-6, \
            f"class_weights must sum to 1, got {self.class_weights.sum()}"
    
    @classmethod
    def from_train_counts(cls, train_class_counts: List[int], num_classes: int = 100):
        """
        Create ReweightedMetrics from training class counts.
        
        Args:
            train_class_counts: List of sample counts per class in training
            num_classes: Number of classes
            
        Returns:
            ReweightedMetrics instance
        """
        train_counts = np.array(train_class_counts)
        class_weights = train_counts / train_counts.sum()
        return cls(class_weights, num_classes)
    
    @classmethod
    def from_json(cls, json_path: str, num_classes: int = 100):
        """
        Load class weights from JSON file.
        
        Args:
            json_path: Path to class_weights.json
            num_classes: Number of classes
            
        Returns:
            ReweightedMetrics instance
        """
        with open(json_path, 'r') as f:
            class_weights = np.array(json.load(f))
        return cls(class_weights, num_classes)
    
    def compute_per_class_accuracy(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-class accuracy.
        
        Args:
            predictions: Array of shape (N,) with predicted class labels
            targets: Array of shape (N,) with true class labels
            
        Returns:
            Tuple of (per_class_accuracy, per_class_correct, per_class_total)
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        per_class_correct = np.zeros(self.num_classes)
        per_class_total = np.zeros(self.num_classes)
        
        for cls in range(self.num_classes):
            cls_mask = (targets == cls)
            per_class_total[cls] = cls_mask.sum()
            
            if per_class_total[cls] > 0:
                per_class_correct[cls] = (predictions[cls_mask] == targets[cls_mask]).sum()
        
        # Avoid division by zero
        per_class_accuracy = np.zeros(self.num_classes)
        for cls in range(self.num_classes):
            if per_class_total[cls] > 0:
                per_class_accuracy[cls] = per_class_correct[cls] / per_class_total[cls]
            else:
                per_class_accuracy[cls] = 0.0  # or np.nan
        
        return per_class_accuracy, per_class_correct, per_class_total
    
    def compute_reweighted_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute accuracy reweighted by training distribution.
        
        This is the CORRECT metric for long-tail datasets.
        
        Example:
            Train: [5000, 500, 50] samples → weights [0.909, 0.091, 0.009]
            Test accuracies: [90%, 85%, 70%]
            
            Normal avg: (90 + 85 + 70) / 3 = 81.67%
            Reweighted: 90×0.909 + 85×0.091 + 70×0.009 = 90.18%
        
        Args:
            predictions: Predicted labels
            targets: True labels
            
        Returns:
            Reweighted accuracy (0 to 1)
        """
        per_class_acc, _, _ = self.compute_per_class_accuracy(predictions, targets)
        reweighted_acc = (per_class_acc * self.class_weights).sum()
        return float(reweighted_acc)
    
    def compute_balanced_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute balanced accuracy (equal weight per class).
        
        This is useful for comparison but NOT the primary metric for long-tail.
        
        Args:
            predictions: Predicted labels
            targets: True labels
            
        Returns:
            Balanced accuracy (0 to 1)
        """
        per_class_acc, _, _ = self.compute_per_class_accuracy(predictions, targets)
        # Only average over classes that have samples
        valid_mask = ~np.isnan(per_class_acc) & (per_class_acc > 0)
        if valid_mask.sum() > 0:
            return float(per_class_acc[valid_mask].mean())
        return 0.0
    
    def compute_head_medium_tail_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        head_classes: Optional[List[int]] = None,
        medium_classes: Optional[List[int]] = None,
        tail_classes: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Compute accuracy for head, medium, and tail classes separately.
        
        Default split (following Cui et al. 2019):
        - Head: top 33% classes (0-32 for 100 classes)
        - Medium: middle 33% classes (33-65 for 100 classes)
        - Tail: bottom 33% classes (66-99 for 100 classes)
        
        For smaller number of classes, adjusts accordingly.
        
        Args:
            predictions: Predicted labels
            targets: True labels
            head_classes: List of head class indices (default: auto-calculated)
            medium_classes: List of medium class indices (default: auto-calculated)
            tail_classes: List of tail class indices (default: auto-calculated)
            
        Returns:
            Dict with 'head_acc', 'medium_acc', 'tail_acc'
        """
        per_class_acc, _, _ = self.compute_per_class_accuracy(predictions, targets)
        
        # Auto-calculate splits based on actual number of classes
        if head_classes is None:
            if self.num_classes >= 100:
                head_classes = list(range(0, 33))
            else:
                # For smaller number of classes, use proportional split
                head_end = max(1, self.num_classes // 3)
                head_classes = list(range(0, head_end))
        
        if medium_classes is None:
            if self.num_classes >= 100:
                medium_classes = list(range(33, 66))
            else:
                head_end = max(1, self.num_classes // 3)
                medium_end = max(head_end + 1, 2 * self.num_classes // 3)
                medium_classes = list(range(head_end, medium_end))
        
        if tail_classes is None:
            if self.num_classes >= 100:
                tail_classes = list(range(66, 100))
            else:
                medium_end = max(1, 2 * self.num_classes // 3)
                tail_classes = list(range(medium_end, self.num_classes))
        
        # Filter to valid indices
        head_classes = [c for c in head_classes if c < self.num_classes]
        medium_classes = [c for c in medium_classes if c < self.num_classes]
        tail_classes = [c for c in tail_classes if c < self.num_classes]
        
        # Compute accuracies
        head_acc = per_class_acc[head_classes].mean() if len(head_classes) > 0 else 0.0
        medium_acc = per_class_acc[medium_classes].mean() if len(medium_classes) > 0 else 0.0
        tail_acc = per_class_acc[tail_classes].mean() if len(tail_classes) > 0 else 0.0
        
        return {
            'head_acc': float(head_acc),
            'medium_acc': float(medium_acc),
            'tail_acc': float(tail_acc)
        }
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        rejection_threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics for long-tail evaluation.
        
        Args:
            predictions: Predicted labels (N,)
            targets: True labels (N,)
            confidences: Confidence scores (N,) - optional
            rejection_threshold: Threshold for rejection - optional
            
        Returns:
            Dictionary with all metrics
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Basic metrics
        per_class_acc, per_class_correct, per_class_total = \
            self.compute_per_class_accuracy(predictions, targets)
        
        results = {
            # Primary metric (MOST IMPORTANT for long-tail)
            'reweighted_accuracy': self.compute_reweighted_accuracy(predictions, targets),
            
            # Secondary metrics
            'balanced_accuracy': self.compute_balanced_accuracy(predictions, targets),
            'overall_accuracy': float((predictions == targets).mean()),
            
            # Per-group metrics
            **self.compute_head_medium_tail_accuracy(predictions, targets),
            
            # Statistics
            'num_samples': len(predictions),
        }
        
        # Add rejection metrics if applicable
        if confidences is not None and rejection_threshold is not None:
            rejection_metrics = self._compute_rejection_metrics(
                predictions, targets, confidences, rejection_threshold
            )
            results.update(rejection_metrics)
        
        return results
    
    def _compute_rejection_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        confidences: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """
        Compute metrics with selective prediction (rejection).
        
        Args:
            predictions: Predicted labels
            targets: True labels
            confidences: Confidence scores
            threshold: Rejection threshold
            
        Returns:
            Dict with rejection-related metrics
        """
        confidences = np.array(confidences)
        accepted_mask = confidences >= threshold
        
        num_accepted = accepted_mask.sum()
        num_total = len(predictions)
        coverage = num_accepted / num_total if num_total > 0 else 0.0
        
        if num_accepted == 0:
            return {
                'coverage': 0.0,
                'rejection_rate': 1.0,
                'accepted_reweighted_acc': 0.0,
                'accepted_balanced_acc': 0.0,
                'rejected_reweighted_acc': 0.0,
            }
        
        # Metrics on accepted samples
        accepted_preds = predictions[accepted_mask]
        accepted_targets = targets[accepted_mask]
        
        accepted_reweighted_acc = self.compute_reweighted_accuracy(
            accepted_preds, accepted_targets
        )
        accepted_balanced_acc = self.compute_balanced_accuracy(
            accepted_preds, accepted_targets
        )
        
        # Metrics on rejected samples
        rejected_mask = ~accepted_mask
        num_rejected = rejected_mask.sum()
        
        if num_rejected > 0:
            rejected_preds = predictions[rejected_mask]
            rejected_targets = targets[rejected_mask]
            rejected_reweighted_acc = self.compute_reweighted_accuracy(
                rejected_preds, rejected_targets
            )
        else:
            rejected_reweighted_acc = 0.0
        
        return {
            'coverage': float(coverage),
            'rejection_rate': float(1.0 - coverage),
            'num_accepted': int(num_accepted),
            'num_rejected': int(num_rejected),
            'accepted_reweighted_acc': float(accepted_reweighted_acc),
            'accepted_balanced_acc': float(accepted_balanced_acc),
            'rejected_reweighted_acc': float(rejected_reweighted_acc),
        }
    
    def print_metrics(self, results: Dict[str, float]):
        """Pretty print metrics."""
        print("\n" + "=" * 60)
        print("EVALUATION METRICS (Long-Tail Reweighted)")
        print("=" * 60)
        
        print("\n🎯 PRIMARY METRIC (for papers):")
        print(f"  Reweighted Accuracy: {results['reweighted_accuracy']:.2%}")
        
        print("\n📊 SECONDARY METRICS:")
        print(f"  Balanced Accuracy:   {results['balanced_accuracy']:.2%}")
        print(f"  Overall Accuracy:    {results['overall_accuracy']:.2%}")
        
        print("\n📈 PER-GROUP ACCURACY:")
        print(f"  Head (0-32):         {results['head_acc']:.2%}")
        print(f"  Medium (33-65):      {results['medium_acc']:.2%}")
        print(f"  Tail (66-99):        {results['tail_acc']:.2%}")
        
        if 'coverage' in results:
            print("\n🚫 REJECTION METRICS:")
            print(f"  Coverage:            {results['coverage']:.2%}")
            print(f"  Rejection Rate:      {results['rejection_rate']:.2%}")
            print(f"  Accepted Samples:    {results['num_accepted']}")
            print(f"  Rejected Samples:    {results['num_rejected']}")
            print(f"  Accepted Reweight Acc: {results['accepted_reweighted_acc']:.2%}")
            print(f"  Rejected Reweight Acc: {results['rejected_reweighted_acc']:.2%}")
        
        print("=" * 60)


def demo_reweighting():
    """
    Demonstrate the difference between reweighted and non-reweighted metrics.
    """
    print("=" * 60)
    print("DEMO: Why Reweighting Matters")
    print("=" * 60)
    
    # Simulate 3 classes with long-tail train distribution
    train_counts = np.array([5000, 500, 50])  # IF=100
    class_weights = train_counts / train_counts.sum()
    
    print(f"\nTrain distribution:")
    print(f"  Class 0 (Head): {train_counts[0]} samples (weight={class_weights[0]:.4f})")
    print(f"  Class 1 (Med):  {train_counts[1]} samples (weight={class_weights[1]:.4f})")
    print(f"  Class 2 (Tail): {train_counts[2]} samples (weight={class_weights[2]:.4f})")
    
    # Simulate balanced test set with different accuracies
    print(f"\nTest set (balanced): 100 samples per class")
    
    # Simulate per-class accuracies
    per_class_acc = np.array([0.90, 0.85, 0.70])  # Head better than tail
    
    print(f"\nPer-class accuracy:")
    print(f"  Class 0: {per_class_acc[0]:.1%}")
    print(f"  Class 1: {per_class_acc[1]:.1%}")
    print(f"  Class 2: {per_class_acc[2]:.1%}")
    
    # Compare metrics
    balanced_acc = per_class_acc.mean()
    reweighted_acc = (per_class_acc * class_weights).sum()
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"❌ Balanced Accuracy (WRONG for long-tail):")
    print(f"   {balanced_acc:.2%} = ({per_class_acc[0]:.0%} + {per_class_acc[1]:.0%} + {per_class_acc[2]:.0%}) / 3")
    print(f"   → Treats all classes equally (not realistic)")
    
    print(f"\n✅ Reweighted Accuracy (CORRECT for long-tail):")
    print(f"   {reweighted_acc:.2%} = {per_class_acc[0]:.0%}×{class_weights[0]:.3f} + "
          f"{per_class_acc[1]:.0%}×{class_weights[1]:.3f} + {per_class_acc[2]:.0%}×{class_weights[2]:.3f}")
    print(f"   → Reflects real-world where head classes dominate")
    
    print(f"\n💡 Key Insight:")
    print(f"   Reweighted accuracy is {reweighted_acc/balanced_acc:.2f}x the balanced accuracy")
    print(f"   because head class (90.1% of data) has high accuracy (90%)")


if __name__ == "__main__":
    # Run demo
    demo_reweighting()
    
    # Example usage with actual data
    print("\n\n" + "=" * 60)
    print("EXAMPLE: Using ReweightedMetrics")
    print("=" * 60)
    
    # Simulate some predictions
    np.random.seed(42)
    num_samples = 1000
    num_classes = 100
    
    # Create fake train distribution (long-tail)
    train_counts = [int(500 * (100 ** (-i/99))) for i in range(num_classes)]
    
    # Create metrics object
    metrics = ReweightedMetrics.from_train_counts(train_counts, num_classes)
    
    # Simulate predictions (better on head, worse on tail)
    targets = np.random.randint(0, num_classes, num_samples)
    predictions = targets.copy()
    
    # Add errors (more errors on tail classes)
    for i in range(num_samples):
        if targets[i] < 33:  # Head
            if np.random.rand() < 0.1:  # 10% error
                predictions[i] = np.random.randint(0, num_classes)
        elif targets[i] < 66:  # Medium
            if np.random.rand() < 0.2:  # 20% error
                predictions[i] = np.random.randint(0, num_classes)
        else:  # Tail
            if np.random.rand() < 0.3:  # 30% error
                predictions[i] = np.random.randint(0, num_classes)
    
    # Compute metrics
    results = metrics.compute_metrics(predictions, targets)
    
    # Print results
    metrics.print_metrics(results)
