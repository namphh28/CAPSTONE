"""
Learning to Reject (LtR) Plugin - Pure Theory Implementation
==============================================================

Implements the plug-in decision rule from LtR paper (Theorem 1):

1. Classifier: h_α(x) = argmax_y (1/α[y] · η̃[y])
2. Rejector:  r(x) = 1{max_y(1/α[y]·η̃[y]) < Σ_y'(1/α[y'] - μ[y'])·η̃[y'] - c}

where:
- η̃(x): mixture posterior from MoE [C]
- α = (α_1, ..., α_C): class reweighting coefficients
- μ = (μ_1, ..., μ_C): normalization vector for threshold
- c: rejection cost (threshold offset)

Group-based simplification (for few samples):
- α[y] = α_{grp(y)} for y in group g
- μ[y] = μ_{grp(y)} for y in group g

Reference: 
    Learning to Reject Meets Long-tail Learning (ICLR 2024)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LtRPluginConfig:
    """Configuration for LtR Plugin."""
    num_classes: int = 100
    num_groups: int = 2
    group_boundaries: List[int] = None  # e.g., [69] for head/tail split
    
    # Optimization mode
    param_mode: str = 'group'  # 'group' or 'class'
    # 'group': α[y] = α_g, μ[y] = μ_g for y in group g (fewer params)
    # 'class': separate α[y], μ[y] for each class (more flexible)
    
    # Grid search ranges
    alpha_grid: List[float] = None  # Reweighting coefficients
    mu_grid: List[float] = None     # Normalization offsets
    cost_grid: List[float] = None   # Rejection costs
    
    # Optimization objective
    objective: str = 'balanced'  # 'balanced' or 'worst'
    
    def __post_init__(self):
        if self.group_boundaries is None:
            self.group_boundaries = [50]
        
        if self.alpha_grid is None:
            # Search around 1.0 (neutral)
            self.alpha_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        if self.mu_grid is None:
            # Search around 0.0 (neutral)
            self.mu_grid = [-0.5, -0.25, 0.0, 0.25, 0.5]
        
        if self.cost_grid is None:
            # Percentile-based approach as suggested by paper
            # Paper suggests using percentile computation to achieve target rejection rate
            # Instead of fixed cost grid, we'll use target rejection rates
            self.target_rejection_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
            self.cost_grid = None  # Will be computed dynamically based on percentiles


class LtRPlugin(nn.Module):
    """
    Pure LtR Plugin with group-based or class-based parameters.
    
    Decision rules (Paper Theorem 1):
        h_α(x) = argmax_y (1/α[y] · η̃[y])
        r(x) = 1 if max_y(1/α[y]·η̃[y]) < Σ_y'(1/α[y'] - μ[y'])·η̃[y'] - c
    
    Interpretation:
    - α[y] > 1: reduce confidence for class y (head penalty)
    - α[y] < 1: boost confidence for class y (tail boosting)
    - μ[y]: adjust threshold based on class distribution
    - c: global rejection cost (higher c = more rejection)
    """
    
    def __init__(self, config: LtRPluginConfig):
        super().__init__()
        self.config = config
        
        # Class-to-group mapping
        class_to_group = torch.zeros(config.num_classes, dtype=torch.long)
        for g_id, boundary in enumerate(config.group_boundaries):
            class_to_group[boundary:] = g_id + 1
        self.register_buffer('class_to_group', class_to_group)
        
        # Parameters (group-based or class-based)
        if config.param_mode == 'group':
            # Group-level parameters (fewer params, more stable)
            self.register_buffer('alpha_group', torch.ones(config.num_groups))
            self.register_buffer('mu_group', torch.zeros(config.num_groups))
        else:
            # Class-level parameters (more flexible, needs more data)
            self.register_buffer('alpha_class', torch.ones(config.num_classes))
            self.register_buffer('mu_class', torch.zeros(config.num_classes))
        
        # Global rejection cost
        self.register_buffer('cost', torch.tensor(0.0))
    
    def get_alpha(self) -> torch.Tensor:
        """Get α vector [C]."""
        if self.config.param_mode == 'group':
            # Map group params to class params
            alpha = self.alpha_group[self.class_to_group]
        else:
            alpha = self.alpha_class
        return alpha
    
    def get_mu(self) -> torch.Tensor:
        """Get μ vector [C]."""
        if self.config.param_mode == 'group':
            # Map group params to class params
            mu = self.mu_group[self.class_to_group]
        else:
            mu = self.mu_class
        return mu
    
    def set_parameters(
        self,
        alpha: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        cost: Optional[float] = None
    ):
        """
        Set optimized parameters.
        
        Args:
            alpha: [G] or [C] depending on param_mode
            mu: [G] or [C] depending on param_mode
            cost: scalar rejection cost
        """
        if alpha is not None:
            if self.config.param_mode == 'group':
                assert len(alpha) == self.config.num_groups
                self.alpha_group = alpha.to(self.alpha_group.device)
            else:
                assert len(alpha) == self.config.num_classes
                self.alpha_class = alpha.to(self.alpha_class.device)
        
        if mu is not None:
            if self.config.param_mode == 'group':
                assert len(mu) == self.config.num_groups
                self.mu_group = mu.to(self.mu_group.device)
            else:
                assert len(mu) == self.config.num_classes
                self.mu_class = mu.to(self.mu_class.device)
        
        if cost is not None:
            self.cost = torch.tensor(cost, device=self.cost.device)
    
    def predict_class(self, mixture_posterior: torch.Tensor) -> torch.Tensor:
        """
        Classifier: h_α(x) = argmax_y (1/α[y]) · η[y] [Paper Formula - Theorem 1]
        
        Args:
            mixture_posterior: [B, C] - p_y(x) from the paper
        
        Returns:
            predictions: [B]
        """
        alpha = self.get_alpha()  # [C]
        
        # CORRECT: Paper uses 1/α, not α
        # Use small epsilon to avoid numerical issues
        eps = 1e-12
        reweighted = mixture_posterior / alpha.unsqueeze(0).clamp(min=eps)  # [B, C]
        
        # Argmax
        predictions = reweighted.argmax(dim=-1)  # [B]
        
        return predictions
    
    def predict_reject(
        self,
        mixture_posterior: torch.Tensor,
        cost: Optional[float] = None
    ) -> torch.Tensor:
        """
        Rejector: r(x) = 1{max_y(1/α[y]·η[y]) < Σ_y'(1/α[y'] - μ[y'])·η[y'] - c} [Paper Formula - Theorem 1]
        
        Args:
            mixture_posterior: [B, C] - p_y(x) from the paper
            cost: optional override for rejection cost
        
        Returns:
            reject: [B] boolean (True = reject)
        """
        alpha = self.get_alpha()  # [C]
        mu = self.get_mu()        # [C]
        
        if cost is None:
            cost = self.cost.item()
        
        # Left side: max_y (1/α[y]) * p_y(x)
        eps = 1e-12
        reweighted = mixture_posterior / alpha.unsqueeze(0).clamp(min=eps)  # [B, C]
        max_reweighted = reweighted.max(dim=-1)[0]  # [B]
        
        # Right side: Σ_y' (1/α[y'] - μ[y']) * p_y'(x) - c
        # CORRECT: Use 1/α - μ as per paper Theorem 1
        threshold_coeff = (1.0 / alpha.clamp(min=eps) - mu).unsqueeze(0)  # [1, C]
        threshold = (threshold_coeff * mixture_posterior).sum(dim=-1) - cost  # [B]
        
        # Rejection decision
        reject = max_reweighted < threshold  # [B]
        
        return reject
    
    def compute_cost_for_target_rejection_rate(
        self,
        mixture_posterior: torch.Tensor,
        target_rejection_rate: float,
        alpha: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute cost c that achieves target rejection rate using percentile approach.
        
        Args:
            mixture_posterior: [B, C]
            target_rejection_rate: target rejection rate (0.0 to 1.0)
            alpha: [G] or [C] (if None, use current alpha)
            mu: [G] or [C] (if None, use current mu)
        
        Returns:
            cost: float that achieves target rejection rate
        """
        if alpha is not None:
            self.set_parameters(alpha=alpha)
        if mu is not None:
            self.set_parameters(mu=mu)
        
        # Compute rejection scores for all samples
        alpha_tensor = self.get_alpha()
        mu_tensor = self.get_mu()
        
        eps = 1e-12
        # CORRECT (Paper Theorem 1): use 1/α for both confidence and threshold terms
        inv_alpha = (1.0 / alpha_tensor.clamp(min=eps))  # [C]
        reweighted = mixture_posterior * inv_alpha.unsqueeze(0)  # [B, C]
        max_reweighted = reweighted.max(dim=-1)[0]  # [B]
        
        # Threshold base: Σ_y (1/α[y] - μ[y]) · p_y(x)
        threshold_coeff = (inv_alpha - mu_tensor).unsqueeze(0)  # [1, C]
        threshold_base = (threshold_coeff * mixture_posterior).sum(dim=-1)  # [B]
        
        # For each sample, compute the cost threshold that would make it rejected
        # reject = max_reweighted < (threshold_base - cost)
        # So cost = threshold_base - max_reweighted for rejected samples
        cost_thresholds = threshold_base - max_reweighted  # [B]
        
        # Sort ascending: reject if t(x) > c. To get rejection rate r, choose c at (1 - r)-quantile
        sorted_costs = torch.sort(cost_thresholds)[0]
        N = len(sorted_costs)
        q = max(0.0, min(1.0, 1.0 - float(target_rejection_rate)))
        # index for (1 - r)-quantile on ascending data
        target_idx = int(round(q * (N - 1)))
        cost = sorted_costs[target_idx].item()
        
        return cost
    
    def forward(
        self,
        mixture_posterior: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass.
        
        Returns:
            dict with 'predictions', 'reject', 'confidence', 'threshold'
        """
        alpha = self.get_alpha()
        mu = self.get_mu()
        
        # Predictions
        predictions = self.predict_class(mixture_posterior)
        
        # Rejection
        reject = self.predict_reject(mixture_posterior)
        
        # Additional info
        eps = 1e-12
        reweighted = mixture_posterior / alpha.unsqueeze(0).clamp(min=eps)
        confidence = reweighted.max(dim=-1)[0]
        threshold_coeff = (1.0 / alpha.clamp(min=eps) - mu).unsqueeze(0)
        threshold = (threshold_coeff * mixture_posterior).sum(dim=-1) - self.cost.item()
        
        return {
            'predictions': predictions,
            'reject': reject,
            'confidence': confidence,
            'threshold': threshold
        }


# ============================================================================
# METRICS
# ============================================================================

def compute_selective_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    reject: torch.Tensor,
    class_to_group: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute selective prediction metrics.
    
    Args:
        predictions: [B] predicted classes
        labels: [B] true labels
        reject: [B] boolean rejection decisions
        class_to_group: [C] mapping from class to group
        sample_weights: [B] optional per-sample weights for reweighting
        beta: [G] optional group weights for weighted objective
    
    Returns:
        dict with:
            - selective_error: error on accepted samples
            - coverage: fraction of accepted samples
            - group_errors: [G] errors per group
            - worst_group_error: max group error
            - weighted_error: beta-weighted error (if beta provided)
    """
    # Accept mask
    accept = ~reject  # [B]
    
    # Handle weights
    if sample_weights is None:
        sample_weights = torch.ones_like(labels, dtype=torch.float32)
    
    # Overall metrics
    if accept.sum() == 0:
        # All rejected
        selective_error = 1.0
        coverage = 0.0
        num_groups = class_to_group.max().item() + 1
        group_errors = [1.0] * num_groups
        worst_group_error = 1.0
    else:
        # Accepted samples
        accept_preds = predictions[accept]
        accept_labels = labels[accept]
        accept_weights = sample_weights[accept]
        
        # Errors (weighted)
        errors = (accept_preds != accept_labels).float()
        selective_error = (errors * accept_weights).sum().item() / accept_weights.sum().item()
        
        # Coverage (weighted)
        coverage = (accept.float() * sample_weights).sum().item() / sample_weights.sum().item()
        
        # Group errors
        groups = class_to_group[accept_labels]  # [B_accept]
        num_groups = class_to_group.max().item() + 1
        
        group_errors = []
        for g in range(num_groups):
            mask_g = (groups == g)
            
            if mask_g.sum() == 0:
                # No samples from this group - worst case error
                group_errors.append(1.0)
            else:
                errors_g = errors[mask_g]
                weights_g = accept_weights[mask_g]
                group_error = (errors_g * weights_g).sum().item() / weights_g.sum().item()
                group_errors.append(group_error)
        
        worst_group_error = max(group_errors)
    
    # Compute weighted error if beta provided
    weighted_error = None
    if beta is not None:
        group_errors_tensor = torch.tensor(group_errors, device=beta.device)
        weighted_error = (beta * group_errors_tensor).sum().item()
    
    result = {
        'selective_error': selective_error,
        'coverage': coverage,
        'group_errors': group_errors,
        'worst_group_error': worst_group_error
    }
    
    if weighted_error is not None:
        result['weighted_error'] = weighted_error
    
    return result


# ============================================================================
# OPTIMIZER
# ============================================================================

@dataclass
class OptimizationResult:
    """Result from grid search."""
    alpha: np.ndarray  # [G] or [C]
    mu: np.ndarray     # [G] or [C]
    cost: float
    
    selective_error: float
    coverage: float
    group_errors: List[float]
    worst_group_error: float
    objective_value: float  # error + cost * (1 - coverage)


class LtRPowerIterOptimizer:
    """
    Power-Iteration Optimizer - Paper Algorithm 1 Implementation
    ============================================================
    
    Implements Algorithm 1: Cost-sensitive Plug-in (CS-plug-in) from paper.
    
    Algorithm:
        For each μ in Λ (multiplier grid):
            α^(0) ← initialize (e.g., prior coverage)
            
            For m = 0 to M-1:  # Power-iteration loop
                # Construct (h, r) with current α^(m)
                ˆα_k ← α^(m)_k * β_k
                h^(m+1)(x) ← argmax_y (1/ˆα[y]) · p_y(x)
                r^(m+1)(x) ← 1 if max_y(...) < threshold
                
                # Update α to match empirical coverage
                α^(m+1)_k ← (1/|S|) * Σ_{(x,y)∈S} 1(y∈G_k, r^(m+1)(x)=0)
            
            Evaluate objective with (h^(M), r^(M))
            Track best result
    
    Advantages over grid search:
        - Adaptive α based on rejection behavior
        - Fewer hyperparameters (only μ, c)
        - Closer to Bayes-optimal solution
        - More efficient (200 configs vs 24k)
    """
    
    def __init__(
        self,
        config: LtRPluginConfig,
        num_iters: int = 20,
        alpha_init_mode: str = 'prior',
        damping: float = 0.0
    ):
        """
        Args:
            config: LtRPluginConfig
            num_iters: M in Algorithm 1 (power-iteration iterations)
            alpha_init_mode: 'prior' (use class prior) or 'uniform' (1/K)
        """
        self.config = config
        self.num_iters = num_iters
        self.alpha_init_mode = alpha_init_mode
        self.damping = damping
    
    def search(
        self,
        plugin: LtRPlugin,
        mixture_posterior: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        verbose: bool = True,
        fixed_mu: Optional[np.ndarray] = None,
        fixed_cost: Optional[float] = None
    ) -> OptimizationResult:
        """
        Power-iteration search following Algorithm 1.
        
        Args:
            plugin: LtRPlugin instance
            mixture_posterior: [B, C]
            labels: [B]
            sample_weights: [B] optional reweighting
            beta: [G] optional group weights for weighted objective
            verbose: print progress
        
        Returns:
            OptimizationResult with best (α, μ, c)
        """
        if verbose:
            print(f"\n[POWER-ITER] Power-Iteration Optimizer (Algorithm 1)")
            print(f"   Mode: {self.config.param_mode}")
            print(f"   Objective: {self.config.objective}")
            print(f"   Power-iter iterations: {self.num_iters}")
        
        best_result = None
        best_objective = float('inf')
        
        # Generate search grid (only μ and c, α will be found by power-iter)
        if fixed_mu is not None and fixed_cost is not None:
            search_grid = [(fixed_mu, fixed_cost)]
        elif fixed_mu is not None:
            # Sweep costs from config, keep μ fixed
            search_grid = [(fixed_mu, cost) for cost in self.config.cost_grid]
        else:
            search_grid = self._generate_search_grid()
        total_configs = len(search_grid)
        
        if verbose:
            print(f"   Total configurations: {total_configs}")
            print(f"   (vs {len(self.config.alpha_grid)**self.config.num_groups * len(self.config.mu_grid)**self.config.num_groups * len(self.config.cost_grid)} in grid search)")
        
        # Outer loop: Grid search over (μ, c)
        for idx, (mu, cost) in enumerate(search_grid):
            # Initialize α^(0)
            alpha = self._initialize_alpha(
                labels,
                plugin.class_to_group,
                sample_weights
            )
            
            # Inner loop: Power-iteration to find optimal α for this (μ, c)
            for m in range(self.num_iters):
                # Set current parameters
                alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=mixture_posterior.device)
                mu_tensor = torch.tensor(mu, dtype=torch.float32, device=mixture_posterior.device)
                plugin.set_parameters(alpha=alpha_tensor, mu=mu_tensor, cost=cost)
                
                # Construct (h^(m+1), r^(m+1)) with current α^(m)
                with torch.no_grad():
                    predictions = plugin.predict_class(mixture_posterior)
                    reject = plugin.predict_reject(mixture_posterior)
                
                # Update α^(m+1) based on empirical coverage
                # First, apply β weighting if provided (for Algorithm 2)
                if beta is not None:
                    # α̂_k = α^{(m)}_k · β_k (from paper)
                    alpha_beta_weighted = alpha * beta.cpu().numpy()
                    # Use β-weighted alpha for classifier and rejector
                    alpha_tensor_beta = torch.tensor(alpha_beta_weighted, dtype=torch.float32, device=mixture_posterior.device)
                    plugin.set_parameters(alpha=alpha_tensor_beta, mu=mu_tensor, cost=cost)
                    
                    # Construct (h^(m+1), r^(m+1)) with β-weighted α
                    with torch.no_grad():
                        predictions = plugin.predict_class(mixture_posterior)
                        reject = plugin.predict_reject(mixture_posterior)
                else:
                    # No β weighting, use original α
                    alpha_tensor_beta = alpha_tensor
                    # predictions and reject already computed above
                
                # Debug: Check rejection rate
                rejection_rate = reject.float().mean().item()
                if verbose and (idx == 0 or idx == len(search_grid) - 1):
                    print(f"     Iter {m+1}: rejection_rate={rejection_rate:.3f}")
                
                # Then update based on empirical coverage
                alpha_new = self._update_alpha_from_coverage(
                    reject,
                    labels,
                    plugin.class_to_group,
                    sample_weights
                )
                
                # Check convergence
                alpha_diff = np.abs(alpha_new - alpha).max()
                
                if verbose and (idx == 0 or idx == len(search_grid) - 1):  # Only show for first and last config
                    print(f"     Iter {m+1}: alpha={alpha} -> {alpha_new}, diff={alpha_diff:.4f}")
                
                # Update alpha for next iteration (with optional damping for stability)
                if self.damping and self.damping > 0.0:
                    alpha = (1.0 - self.damping) * alpha + self.damping * alpha_new
                else:
                    alpha = alpha_new
                
                # Early stopping if converged
                if alpha_diff < 1e-4:
                    if verbose and (idx == 0 or idx == len(search_grid) - 1):
                        print(f"     Converged after {m+1} iterations")
                    break
            
            # Evaluate final (h^(M), r^(M)) after M iterations
            # Set final α
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=mixture_posterior.device)
            plugin.set_parameters(alpha=alpha_tensor, mu=mu_tensor, cost=cost)
            
            with torch.no_grad():
                predictions = plugin.predict_class(mixture_posterior)
                reject = plugin.predict_reject(mixture_posterior)
            
            # Paper metrics for balanced/worst should not use class reweighting
            metrics = compute_selective_metrics(
                predictions, labels, reject,
                plugin.class_to_group,
                sample_weights=None,
                beta=beta
            )
            
            # Compute objective value
            if beta is not None and 'weighted_error' in metrics:
                # Use weighted objective when beta provided (for Algorithm 2)
                error = metrics['weighted_error']
            elif self.config.objective == 'balanced':
                error = np.mean(metrics['group_errors'])
            else:  # worst
                error = metrics['worst_group_error']
            
            objective_value = error + cost * (1.0 - metrics['coverage'])
            
            # Track best
            if objective_value < best_objective:
                best_objective = objective_value
                best_result = OptimizationResult(
                    alpha=alpha,
                    mu=mu,
                    cost=cost,
                    selective_error=metrics['selective_error'],
                    coverage=metrics['coverage'],
                    group_errors=metrics['group_errors'],
                    worst_group_error=metrics['worst_group_error'],
                    objective_value=objective_value
                )
            
            # Progress
            if verbose and (idx + 1) % max(1, total_configs // 10) == 0:
                print(f"   Progress: {idx+1}/{total_configs} "
                      f"(best obj={best_objective:.4f}, "
                      f"alpha={best_result.alpha if best_result else None})")
        
        if verbose:
            print(f"\n[SUCCESS] Best configuration found:")
            print(f"   alpha = {best_result.alpha} (found by power-iter)")
            print(f"   mu = {best_result.mu}")
            print(f"   c = {best_result.cost:.3f}")
            print(f"   Objective: {best_result.objective_value:.4f}")
            print(f"   Selective error: {best_result.selective_error:.4f}")
            print(f"   Coverage: {best_result.coverage:.3f}")
            print(f"   Group errors: {[f'{e:.4f}' for e in best_result.group_errors]}")
        
        return best_result
    
    def _initialize_alpha(
        self,
        labels: torch.Tensor,
        class_to_group: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Initialize α^(0) based on prior coverage or uniform.
        
        α_k should reflect expected coverage (proportion of samples from group k).
        
        Args:
            labels: [B]
            class_to_group: [C]
            sample_weights: [B] optional
        
        Returns:
            alpha: [G] initial values
        """
        num_groups = class_to_group.max().item() + 1
        alpha = np.zeros(num_groups)
        
        if sample_weights is None:
            sample_weights = torch.ones_like(labels, dtype=torch.float32)
        
        if self.alpha_init_mode == 'prior':
            # α_k = K * proportion of samples from group k (weighted)
            # This ensures α ∈ (0, K) as required by paper
            groups = class_to_group[labels]  # [B]
            
            for g in range(num_groups):
                mask_g = (groups == g)
                if mask_g.sum() > 0:
                    # Calculate proportion first
                    proportion = (mask_g.float() * sample_weights).sum().item() / sample_weights.sum().item()
                    # Scale by K to get α ∈ (0, K)
                    alpha[g] = num_groups * proportion
                else:
                    alpha[g] = 1.0  # fallback: α = 1 (within (0, K))
        
        else:  # uniform
            # α_k = K * (1/K) = 1 for all groups (within (0, K))
            alpha = np.ones(num_groups)
        
        # Ensure α > 0 and α < K (avoid division by zero and stay in range)
        alpha = np.maximum(alpha, 1e-4)
        alpha = np.minimum(alpha, num_groups - 1e-4)
        
        return alpha
    
    def _update_alpha_from_coverage(
        self,
        reject: torch.Tensor,
        labels: torch.Tensor,
        class_to_group: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        current_alpha: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Update α based on empirical coverage (Algorithm 1, Line 7).
        
        According to paper: α_k = α^{(m)}_k * β_k
        Then: α^(m+1)_k = (1/|S|) * Σ_{(x,y)∈S} 1(y∈G_k, r^(m+1)(x)=0)
        
        Interpretation: α_k = proportion of group k samples that are ACCEPTED.
        
        Args:
            reject: [B] rejection decisions
            labels: [B]
            class_to_group: [C]
            sample_weights: [B] optional
            beta: [G] group weights (optional, for Algorithm 2)
            current_alpha: [G] current alpha values (optional, for Algorithm 2)
        
        Returns:
            alpha: [G] updated values
        """
        num_groups = class_to_group.max().item() + 1
        alpha = np.zeros(num_groups)
        
        if sample_weights is None:
            sample_weights = torch.ones_like(labels, dtype=torch.float32)
        
        # Accept mask
        accept = ~reject  # [B]
        
        if accept.sum() == 0:
            # All rejected - fallback to uniform
            return np.ones(num_groups) / num_groups
        
        # Groups of accepted samples
        accept_groups = class_to_group[labels[accept]]  # [B_accept]
        accept_weights = sample_weights[accept]         # [B_accept]
        
        # Compute coverage per group
        for g in range(num_groups):
            # Count samples from group g that are accepted
            group_mask = (class_to_group[labels] == g)  # All samples in group g
            accepted_in_group = group_mask & accept  # Samples in group g that are accepted
            
            if accepted_in_group.sum() > 0:
                # α_k^(m+1) = K * P(y ∈ G_k, r^(m+1)(x) = 0)
                # This is K times the proportion of ALL samples that are in group g AND accepted
                empirical_coverage = accepted_in_group.sum().float().item() / len(labels)
                alpha[g] = num_groups * empirical_coverage  # Multiply by K (num_groups)
            else:
                alpha[g] = 1e-4  # small value for empty groups
            
            # Apply β weighting if provided (for Algorithm 2)
            if beta is not None and current_alpha is not None:
                # α_k = α^{(m)}_k * β_k (from paper) - but we need to apply this BEFORE empirical update
                # This is handled in the calling code, not here
                pass
        
        # Ensure α > 0
        alpha = np.maximum(alpha, 1e-4)
        
        return alpha
    
    def _generate_search_grid(self) -> List[Tuple[np.ndarray, float]]:
        """
        Generate search grid over (μ, c) only.

        Paper Appendix E.1: For K=2, re-parameterize to a single
        multiplier λ = μ_tail − μ_head. We can fix μ_head = 0 and
        set μ_tail = λ without loss of generality, searching λ on a 1D grid.

        Returns:
            List of (mu, cost) tuples
        """
        grid = []
        
        if self.config.num_groups == 2:
            # 1D search on λ with μ = [0, λ]
            # Preserve backward compatibility with existing mu_grid values
            lambda_grid = self.config.mu_grid
            for lam in lambda_grid:
                for cost in self.config.cost_grid:
                    mu = np.array([0.0, lam])
                    grid.append((mu, cost))
        else:
            # General case
            raise NotImplementedError(f"Power-iter for {self.config.num_groups} groups not implemented")
        
        return grid


class LtRGridSearchOptimizer:
    """
    Grid search optimizer for LtR Plugin (baseline, less efficient).
    
    Searches over (α, μ, c) to minimize objective:
        - Balanced: mean(group_errors) + c * (1 - coverage)
        - Worst-group: max(group_errors) + c * (1 - coverage)
        - Weighted: Σ_k β_k * group_errors[k] + c * (1 - coverage) (if beta provided)
    
    Note: This is a BASELINE. Use LtRPowerIterOptimizer for paper-compliant method.
    """
    
    def __init__(self, config: LtRPluginConfig):
        self.config = config
    
    def search(
        self,
        plugin: LtRPlugin,
        mixture_posterior: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Grid search over (α, μ, c).
        
        Args:
            plugin: LtRPlugin instance
            mixture_posterior: [B, C]
            labels: [B]
            sample_weights: [B] optional reweighting
            beta: [G] optional group weights for weighted objective
            verbose: print progress
        
        Returns:
            OptimizationResult with best parameters
        """
        if verbose:
            print(f"\n🔍 Grid Search ({self.config.param_mode} mode)")
        
        best_result = None
        best_objective = float('inf')
        
        # Generate parameter grid
        if self.config.param_mode == 'group':
            num_params = self.config.num_groups
            param_grid = self._generate_group_grid()
        else:
            num_params = self.config.num_classes
            param_grid = self._generate_class_grid()
        
        total_configs = len(param_grid)
        
        if verbose:
            print(f"   Total configurations: {total_configs}")
            print(f"   Objective: {self.config.objective}")
        
        # Evaluate each configuration
        for idx, (alpha, mu, cost) in enumerate(param_grid):
            # Set parameters
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=mixture_posterior.device)
            mu_tensor = torch.tensor(mu, dtype=torch.float32, device=mixture_posterior.device)
            plugin.set_parameters(alpha=alpha_tensor, mu=mu_tensor, cost=cost)
            
            # Forward pass
            with torch.no_grad():
                predictions = plugin.predict_class(mixture_posterior)
                reject = plugin.predict_reject(mixture_posterior)
            
            # Compute metrics
            metrics = compute_selective_metrics(
                predictions, labels, reject,
                plugin.class_to_group,
                sample_weights,
                beta
            )
            
            # Objective value
            if beta is not None and 'weighted_error' in metrics:
                # Use weighted objective when beta provided (for Algorithm 2)
                error = metrics['weighted_error']
            elif self.config.objective == 'balanced':
                error = np.mean(metrics['group_errors'])
            else:  # worst
                error = metrics['worst_group_error']
            
            objective_value = error + cost * (1.0 - metrics['coverage'])
            
            # Track best
            if objective_value < best_objective:
                best_objective = objective_value
                best_result = OptimizationResult(
                    alpha=alpha,
                    mu=mu,
                    cost=cost,
                    selective_error=metrics['selective_error'],
                    coverage=metrics['coverage'],
                    group_errors=metrics['group_errors'],
                    worst_group_error=metrics['worst_group_error'],
                    objective_value=objective_value
                )
            
            # Progress
            if verbose and (idx + 1) % max(1, total_configs // 10) == 0:
                print(f"   Progress: {idx+1}/{total_configs} "
                      f"(best obj={best_objective:.4f})")
        
        if verbose:
            print(f"\n[SUCCESS] Best configuration found:")
            print(f"   α = {best_result.alpha}")
            print(f"   mu = {best_result.mu}")
            print(f"   c = {best_result.cost:.3f}")
            print(f"   Objective: {best_result.objective_value:.4f}")
            print(f"   Selective error: {best_result.selective_error:.4f}")
            print(f"   Coverage: {best_result.coverage:.3f}")
            print(f"   Group errors: {[f'{e:.4f}' for e in best_result.group_errors]}")
        
        return best_result
    
    def _generate_group_grid(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate grid for group-based parameters."""
        grid = []
        
        # For 2 groups (head/tail)
        if self.config.num_groups == 2:
            for alpha_head in self.config.alpha_grid:
                for alpha_tail in self.config.alpha_grid:
                    for mu_head in self.config.mu_grid:
                        for mu_tail in self.config.mu_grid:
                            for cost in self.config.cost_grid:
                                alpha = np.array([alpha_head, alpha_tail])
                                mu = np.array([mu_head, mu_tail])
                                grid.append((alpha, mu, cost))
        else:
            # General case (not implemented yet)
            raise NotImplementedError(f"Grid search for {self.config.num_groups} groups not implemented")
        
        return grid
    
    def _generate_class_grid(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate grid for class-based parameters (too expensive for 100 classes)."""
        raise NotImplementedError("Class-level grid search too expensive for 100 classes. Use group mode.")


# ============================================================================
# RC CURVE COMPUTATION
# ============================================================================

class RCCurveComputer:
    """
    Compute Risk-Coverage (RC) curve by sweeping rejection cost c.
    
    For fixed (α, μ), vary c from low to high to get different (coverage, error) points.
    """
    
    def __init__(self, config: LtRPluginConfig):
        self.config = config
    
    def compute_rc_curve(
        self,
        plugin: LtRPlugin,
        mixture_posterior: torch.Tensor,
        labels: torch.Tensor,
        alpha: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        cost_grid: Optional[List[float]] = None,
        sample_weights: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Compute RC curve by sweeping cost c.
        
        Args:
            plugin: LtRPlugin instance
            mixture_posterior: [B, C]
            labels: [B]
            alpha: [G] or [C] (if None, use plugin's current alpha)
            mu: [G] or [C] (if None, use plugin's current mu)
            cost_grid: list of costs to sweep (if None, use config)
            sample_weights: [B] optional
            beta: [G] optional group weights for weighted objective
        
        Returns:
            dict with RC curve data
        """
        # Set fixed (α, μ)
        if alpha is not None:
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=mixture_posterior.device)
            plugin.set_parameters(alpha=alpha_tensor)
        
        if mu is not None:
            mu_tensor = torch.tensor(mu, dtype=torch.float32, device=mixture_posterior.device)
            plugin.set_parameters(mu=mu_tensor)
        
        # Cost grid - use percentile-based approach if available
        if cost_grid is None:
            if hasattr(self.config, 'target_rejection_rates') and self.config.target_rejection_rates is not None:
                # Percentile-based approach: compute costs for target rejection rates
                cost_grid = []
                for target_rejection_rate in self.config.target_rejection_rates:
                    cost = plugin.compute_cost_for_target_rejection_rate(
                        mixture_posterior, target_rejection_rate
                    )
                    cost_grid.append(cost)
            else:
                cost_grid = np.linspace(0.0, 1.0, 100)
        
        # Sweep costs
        rejection_rates = []
        selective_errors = []
        group_errors_list = []
        balanced_errors = []
        
        for cost in cost_grid:
            # Set cost
            plugin.set_parameters(cost=cost)
            
            # Forward pass
            with torch.no_grad():
                predictions = plugin.predict_class(mixture_posterior)
                reject = plugin.predict_reject(mixture_posterior)
            
            # Metrics
            metrics = compute_selective_metrics(
                predictions, labels, reject,
                plugin.class_to_group,
                sample_weights,
                beta  # beta is None for RC curve computation
            )
            
            rejection_rate = 1.0 - metrics['coverage']
            rejection_rates.append(rejection_rate)
            selective_errors.append(metrics['selective_error'])
            group_errors_list.append(metrics['group_errors'])
            # Balanced error per paper = mean of per-group errors
            balanced_errors.append(float(np.mean(metrics['group_errors'])))
        
        # Convert to arrays
        rejection_rates = np.array(rejection_rates)
        selective_errors = np.array(selective_errors)
        balanced_errors = np.array(balanced_errors)
        
        # CRITICAL: Sort by rejection rate (ascending) for correct AURC calculation
        # RC curve MUST go from low rejection (0) to high rejection (1)
        sort_idx = np.argsort(rejection_rates)
        rejection_rates = rejection_rates[sort_idx]
        selective_errors = selective_errors[sort_idx]
        balanced_errors = balanced_errors[sort_idx]
        
        # Compute AURC (now with properly sorted data)
        # For paper-compliant AURC on balanced error
        aurc_balanced = np.trapz(balanced_errors, rejection_rates)
        # Keep overall AURC as well for reference
        aurc_overall = np.trapz(selective_errors, rejection_rates)
        
        return {
            'rejection_rates': rejection_rates,
            'selective_errors': selective_errors,
            'balanced_errors': balanced_errors,
            'group_errors_list': group_errors_list,
            'aurc': aurc_balanced,
            'aurc_overall': aurc_overall,
            'cost_grid': cost_grid
        }


# ============================================================================
# ALGORITHM 2: WORST-GROUP PLUG-IN (Exponentiated Gradient)
# ============================================================================

class LtRWorstGroupOptimizer:
    """
    Algorithm 2: Worst-group Plug-in from paper.
    
    Optimizes worst-group error using exponentiated gradient ascent on β.
    
    Algorithm:
        β^(0) ← uniform (1/K for each group)
        
        For t = 0 to T-1:
            # Solve cost-sensitive problem with current β^(t)
            (h^(t), r^(t)) ← Algorithm1(β^(t), c)
            
            # Compute group errors
            e_k^(t) ← empirical_error(group_k, h^(t), r^(t))
            
            # Update β using exponentiated gradient
            β^(t+1)_k ∝ β^(t)_k * exp(ξ * e_k^(t))  # Up-weight high-error groups
            β^(t+1) ← β^(t+1) / ||β^(t+1)||_1       # Normalize
        
        Return (h^(T), r^(T)) with lowest max_k e_k
    
    Intuition: 
        - Groups with high error get higher β → more weight in next iteration
        - Eventually converges to balanced worst-group error
    """
    
    def __init__(
        self,
        config: LtRPluginConfig,
        num_outer_iters: int = 25,  # Paper F.3: T=25
        learning_rate: float = 1.0,  # Paper F.3: ξ=1
        use_power_iter: bool = True
    ):
        """
        Args:
            config: LtRPluginConfig
            num_outer_iters: T in Algorithm 2 (outer iterations)
            learning_rate: ξ (learning rate for exp gradient)
            use_power_iter: if True, use Algorithm 1; else use grid search
        """
        self.config = config
        self.num_outer_iters = num_outer_iters
        self.learning_rate = learning_rate
        self.use_power_iter = use_power_iter
        
        # Create inner optimizer
        if use_power_iter:
            self.inner_optimizer = LtRPowerIterOptimizer(config)
        else:
            self.inner_optimizer = LtRGridSearchOptimizer(config)
    
    def search(
        self,
        plugin: LtRPlugin,
        mixture_posterior_s1: torch.Tensor,
        labels_s1: torch.Tensor,
        mixture_posterior_s2: torch.Tensor,
        labels_s2: torch.Tensor,
        sample_weights_s1: Optional[torch.Tensor] = None,
        sample_weights_s2: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Worst-group optimization with exponentiated gradient.
        
        According to paper Algorithm 2:
        - S1: used for inner optimization (Algorithm 1)
        - S2: used for estimating ê_k (group errors)
        
        Args:
            plugin: LtRPlugin instance
            mixture_posterior_s1: [B1, C] - S1 data for inner optimization
            labels_s1: [B1] - S1 labels
            mixture_posterior_s2: [B2, C] - S2 data for estimating ê_k
            labels_s2: [B2] - S2 labels
            sample_weights_s1: [B1] optional - S1 sample weights
            sample_weights_s2: [B2] optional - S2 sample weights
            verbose: print progress
        
        Returns:
            OptimizationResult with best (α, μ, c) for worst-group
        """
        if verbose:
            print(f"\n🎯 Worst-Group Optimizer (Algorithm 2)")
            print(f"   Outer iterations: {self.num_outer_iters}")
            print(f"   Learning rate: {self.learning_rate}")
            print(f"   Inner optimizer: {'Power-iter' if self.use_power_iter else 'Grid search'}")
        
        # Initialize β^(0) = uniform
        num_groups = self.config.num_groups
        beta = np.ones(num_groups) / num_groups
        
        best_result = None
        best_worst_error = float('inf')
        
        # Outer loop: Exponentiated gradient on β
        for t in range(self.num_outer_iters):
            if verbose:
                print(f"\n--- Iteration {t+1}/{self.num_outer_iters} ---")
                print(f"   Current β = {beta}")
            
            # Solve Algorithm 1 with current β^(t) weights on S1
            # This is the CORRECT implementation according to paper Algorithm 2
            result = self.inner_optimizer.search(
                plugin, mixture_posterior_s1, labels_s1, sample_weights_s1,
                beta=torch.tensor(beta, dtype=torch.float32, device=mixture_posterior_s1.device),
                verbose=False  # Suppress inner logs
            )
            
            if verbose:
                print(f"   Inner optimization result: α = {result.alpha}, μ = {result.mu}, cost = {result.cost:.4f}")
            
            # Compute group errors e_k^(t) on S2 using the (h^(t), r^(t)) from S1
            # This follows the paper's requirement to use S2 for estimating ê_k
            plugin.set_parameters(
                alpha=torch.tensor(result.alpha, dtype=torch.float32, device=mixture_posterior_s1.device),
                mu=torch.tensor(result.mu, dtype=torch.float32, device=mixture_posterior_s1.device),
                cost=result.cost
            )
            
            with torch.no_grad():
                predictions_s2 = plugin.predict_class(mixture_posterior_s2)
                reject_s2 = plugin.predict_reject(mixture_posterior_s2, cost=result.cost)
                
                # Compute group errors on S2, only on accepted samples
                group_errors = self._compute_group_errors_on_s2(
                    plugin, mixture_posterior_s2, labels_s2, sample_weights_s2
                )
                worst_error = np.max(group_errors)
                
                if verbose:
                    print(f"   Group errors: {[f'{e:.4f}' for e in group_errors]}")
                    print(f"   Worst error: {worst_error:.4f}")
                
                # Track best
                if worst_error < best_worst_error:
                    best_worst_error = worst_error
                    best_result = result
                    if verbose:
                        print(f"   ✓ New best worst-error: {worst_error:.4f}")
                
                # Update β^(t+1) using exponentiated gradient
                # β^(t+1)_k ∝ β^(t)_k * exp(ξ * e_k^(t))
                beta_old = beta.copy()
                beta = beta * np.exp(self.learning_rate * group_errors)
                
                # Normalize to simplex
                beta = beta / beta.sum()
                
                if verbose:
                    beta_change = np.abs(beta - beta_old).max()
                    print(f"   β change: {beta_change:.6f}")
                
                # Early stopping if β converges (change < threshold)
                if beta_change < 1e-6:
                    if verbose:
                        print(f"   Early stopping: β converged at iteration {t+1}")
                    break
        
        if verbose:
            print(f"\n✅ Best worst-group configuration:")
            print(f"   α = {best_result.alpha}")
            print(f"   mu = {best_result.mu}")
            print(f"   c = {best_result.cost:.3f}")
            print(f"   Worst-group error: {best_result.worst_group_error:.4f}")
            print(f"   Group errors: {[f'{e:.4f}' for e in best_result.group_errors]}")
        
        return best_result
    
    def _compute_group_errors_on_s2(
        self, 
        plugin: LtRPlugin, 
        mixture_posterior_s2: torch.Tensor, 
        labels_s2: torch.Tensor, 
        sample_weights_s2: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute group errors ê_k on S2, only on accepted samples
        ê_k(h,r) = Σ_{(x,y)∈S2} 1{y ≠ h(x), y ∈ G_k, r(x) = 0} / Σ_{(x,y)∈S2} 1{y ∈ G_k, r(x) = 0}
        """
        with torch.no_grad():
            predictions = plugin.predict_class(mixture_posterior_s2)
            reject = plugin.predict_reject(mixture_posterior_s2)
            
            group_errors = []
            for g in range(plugin.config.num_groups):
                # Find samples in group g
                group_mask = (plugin.class_to_group[labels_s2] == g)
                
                # Only consider accepted samples (r(x) = 0)
                accepted_in_group = group_mask & (~reject)
                
                if accepted_in_group.sum() > 0:
                    # Count errors in accepted samples of group g
                    errors_in_group = (predictions[accepted_in_group] != labels_s2[accepted_in_group]).sum().float()
                    total_accepted_in_group = accepted_in_group.sum().float()
                    
                    # ê_k = errors / accepted_samples_in_group
                    group_error = (errors_in_group / total_accepted_in_group).item()
                else:
                    # No accepted samples in group g → use smoothing to avoid NaN
                    # Use total samples in group as denominator (with smoothing)
                    total_in_group = group_mask.sum().float()
                    if total_in_group > 0:
                        # Assume worst case: all samples in group would be errors
                        group_error = 1.0  # Max error when no accepted samples
                    else:
                        group_error = 0.5  # Neutral error when group is empty
                
                group_errors.append(group_error)
            
            return np.array(group_errors)

