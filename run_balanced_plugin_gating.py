#!/usr/bin/env python3
"""
Standalone Balanced Plug-in with Gating (3 Experts) per "Learning to Reject Meets Long-Tail Learning"
=======================================================================================================

- Loads CIFAR-100-LT/iNaturalist 2018 splits and expert logits
- Loads trained gating network to combine experts
- Builds head/tail groups from train class counts (tail <= 20 samples)
- Implements Theorem 1 decision rules (classifier and rejector)
- Implements Algorithm 1 (power-iteration) over α and 1D λ grid for μ
- Runs theory-compliant cost sweep: optimize (α, μ) per cost, one RC point per cost
- Evaluates on test; computes balanced error RC and AURC; saves JSON and plots

Usage:
    python run_balanced_plugin_gating.py --dataset cifar100_lt_if100
    python run_balanced_plugin_gating.py --dataset inaturalist2018

Inputs (expected existing):
- Splits dir: ./data/{dataset_name}_splits/
- Logits dir: ./outputs/logits/{dataset_name}/{expert_name}/{split}_logits.pt
- Gating checkpoint: ./checkpoints/gating_map/{dataset_name}/final_gating.pth
- Targets (if available): ./outputs/logits/{dataset_name}/{expert_name}/{split}_targets.pt

Outputs:
- results/ltr_plugin/{dataset_name}/ltr_plugin_gating_balanced.json
- results/ltr_plugin/{dataset_name}/ltr_rc_curves_balanced_gating_test.png
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ================================
# Config
# ================================
@dataclass
class Config:
    dataset_name: str = "cifar100_lt_if100"
    splits_dir: str = "./data/cifar100_lt_if100_splits_fixed"
    logits_dir: str = "./outputs/logits/cifar100_lt_if100"
    gating_checkpoint: str = (
        "./checkpoints/gating_map/cifar100_lt_if100/final_gating.pth"
    )
    results_dir: str = "./results/ltr_plugin/cifar100_lt_if100"

    expert_names: List[str] = field(
        default_factory=lambda: [
            "ce_baseline",
            "logitadjust_baseline",
            "balsoftmax_baseline",
        ]
    )

    num_classes: int = 100
    num_groups: int = 2

    # Tail definition per paper: tail if train count <= 20
    tail_threshold: int = 20

    # Optimizer settings - extended range including paper values {1, 6, 11}
    mu_lambda_grid: List[float] = (
        -5.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        5.0,
        6.0,
        8.0,
        11.0,
        15.0,
        20.0,
    )
    power_iter_iters: int = 20  # More iterations for convergence
    power_iter_damping: float = 0.5  # Higher damping for stability

    # Cost sweep (theory-compliant): one RC point per cost
    cost_sweep: List[float] = ()  # unused when target_rejections set
    # Target rejection grid to match paper plots exactly
    target_rejections: List[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

    seed: int = 42


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Dataset configurations
DATASET_CONFIGS = {
    "cifar100_lt_if100": {
        "splits_dir": "./data/cifar100_lt_if100_splits_fixed",
        "logits_dir": "./outputs/logits/cifar100_lt_if100",
        "gating_checkpoint": "./checkpoints/gating_map/cifar100_lt_if100/final_gating.pth",
        "results_dir": "./results/ltr_plugin/cifar100_lt_if100",
        "expert_names": ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"],
        "num_classes": 100,
        "num_groups": 2,
    },
    "inaturalist2018": {
        "splits_dir": "./data/inaturalist2018_splits",
        "logits_dir": "./outputs/logits/inaturalist2018",
        "gating_checkpoint": "./checkpoints/gating_map/inaturalist2018/final_gating.pth",
        "results_dir": "./results/ltr_plugin/inaturalist2018",
        "expert_names": ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"],
        "num_classes": 8142,
        "num_groups": 2,
    }
}


def setup_config(dataset_name: str) -> Config:
    """Setup Config based on dataset selection."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}")
    
    ds_config = DATASET_CONFIGS[dataset_name]
    
    # Create Config with dataset-specific settings
    config = Config(
        dataset_name=dataset_name,
        splits_dir=ds_config["splits_dir"],
        logits_dir=ds_config["logits_dir"],
        gating_checkpoint=ds_config["gating_checkpoint"],
        results_dir=ds_config["results_dir"],
        expert_names=ds_config["expert_names"],
        num_classes=ds_config["num_classes"],
        num_groups=ds_config["num_groups"],
    )
    
    return config


# Default config (will be overridden by main)
CFG = Config()


# ================================
# IO helpers
# ================================


def load_expert_logits(
    expert_names: List[str], split: str, device: str = DEVICE
) -> torch.Tensor:
    """Load logits from all experts and stack them."""
    logits_list = []

    print(f"Loading logits for {len(expert_names)} experts: {expert_names}")
    for expert_name in expert_names:
        path = Path(CFG.logits_dir) / expert_name / f"{split}_logits.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing logits: {path}")
        logits = torch.load(path, map_location=device).float()
        logits_list.append(logits)
        print(f"  ✓ Loaded {expert_name}: {logits.shape}")

    # Stack: [E, N, C] -> transpose to [N, E, C]
    logits = torch.stack(logits_list, dim=0).transpose(0, 1)
    print(
        f"✓ Stacked expert logits: {logits.shape} (should be [N, {len(expert_names)}, {CFG.num_classes}])"
    )
    return logits


def load_labels(split: str, device: str = DEVICE) -> torch.Tensor:
    # Prefer saved targets alongside logits
    cand = Path(CFG.logits_dir) / CFG.expert_names[0] / f"{split}_targets.pt"
    if cand.exists():
        t = torch.load(cand, map_location=device)
        if isinstance(t, torch.Tensor):
            return t.to(device=device, dtype=torch.long)

    # Fallback: load from JSON targets file (for iNaturalist) or CIFAR-style loading
    targets_file = Path(CFG.splits_dir) / f"{split}_targets.json"
    if targets_file.exists():
        # iNaturalist: load targets from JSON
        with open(targets_file, "r", encoding="utf-8") as f:
            targets = json.load(f)
        return torch.tensor(targets, dtype=torch.long, device=device)
    
    # Fallback: reconstruct from CIFAR100 and indices
    import torchvision

    indices_file = Path(CFG.splits_dir) / f"{split}_indices.json"
    with open(indices_file, "r", encoding="utf-8") as f:
        indices = json.load(f)
    is_train = split in ("expert", "gating", "train")
    ds = torchvision.datasets.CIFAR100(root="./data", train=is_train, download=False)
    labels = torch.tensor(
        [ds.targets[i] for i in indices], dtype=torch.long, device=device
    )
    return labels


def load_class_weights(device: str = DEVICE) -> torch.Tensor:
    """Load inverse class weights for importance weighting to re-weight test set to training distribution."""
    counts_path = Path(CFG.splits_dir) / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(CFG.num_classes)]

    counts = np.array(class_counts, dtype=np.float64)
    total_train = counts.sum()

    # CORRECTED: Calculate inverse weights to re-weight test set to training distribution
    # Test set is balanced (1/num_classes per class), training set is long-tail
    train_probs = counts / total_train
    test_probs = np.ones(CFG.num_classes) / CFG.num_classes  # Balanced test set

    # Importance weights = train_probs / test_probs = train_probs * num_classes
    # This up-weights head classes and down-weights tail classes
    weights = train_probs * CFG.num_classes

    print(
        f"Training distribution: head={train_probs[0]:.6f}, tail={train_probs[-1]:.6f}"
    )
    print(f"Test distribution (balanced): {1.0 / CFG.num_classes:.6f}")
    print(f"Importance weights: head={weights[0]:.6f}, tail={weights[-1]:.6f}")
    print(f"Weight ratio (head/tail): {weights[0] / weights[-1]:.1f}x")

    return torch.tensor(weights, dtype=torch.float32, device=device)


def load_gating_network(device: str = DEVICE):  # Returns GatingNetwork
    """Load trained gating network."""
    from src.models.gating_network_map import GatingNetwork, GatingMLP
    from src.models.gating import GatingFeatureBuilder

    num_experts = len(CFG.expert_names)
    num_classes = CFG.num_classes

    print(f"Loading gating network for {num_experts} experts: {CFG.expert_names}")

    gating = GatingNetwork(
        num_experts=num_experts, num_classes=num_classes, routing="dense"
    ).to(device)

    # Match compact feature setup used during training (GatingFeatureBuilder: D = 7*E + 3)
    compact_dim = 7 * num_experts + 3
    gating.mlp = GatingMLP(
        input_dim=compact_dim,
        num_experts=num_experts,
        hidden_dims=[256, 128],
        dropout=0.1,
        activation='relu',
    ).to(device)

    checkpoint_path = Path(CFG.gating_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing gating checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Verify checkpoint matches expected number of experts
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        # Check if mlp output layer has correct number of experts
        if (
            "mlp.mlp.8.weight" in state_dict
        ):  # Last layer of MLP (assuming 2 hidden layers)
            mlp_output_dim = state_dict["mlp.mlp.8.weight"].shape[0]
            if mlp_output_dim != num_experts:
                raise ValueError(
                    f"Checkpoint expects {mlp_output_dim} experts but config has {num_experts} experts. "
                    f"Please check expert_names: {CFG.expert_names}"
                )

    gating.load_state_dict(checkpoint["model_state_dict"])
    gating.eval()

    print(f"✓ Loaded gating network from: {checkpoint_path}")
    print(f"✓ Gating network configured for {num_experts} experts: {CFG.expert_names}")
    return gating


def compute_mixture_posterior(
    expert_logits: torch.Tensor,
    gating_net,
    device: str = DEVICE,  # gating_net: GatingNetwork
) -> torch.Tensor:
    """Compute mixture posterior using gating network."""
    # expert_logits: [N, E, C]
    with torch.no_grad():
        # Convert logits to posteriors (for mixture) and build compact gating features
        expert_posteriors = F.softmax(expert_logits, dim=-1)  # [N, E, C]
        from src.models.gating import GatingFeatureBuilder
        feat_builder = GatingFeatureBuilder()
        features = feat_builder(expert_logits)                # [N, 7*E+3]

        # Verify shape matches expected number of experts
        num_experts_logits = expert_logits.shape[1]
        num_experts_config = len(CFG.expert_names)
        if num_experts_logits != num_experts_config:
            raise ValueError(
                f"Mismatch: logits have {num_experts_logits} experts but config expects {num_experts_config} experts"
            )

        # Get gating weights via mlp + router on compact features
        gating_logits = gating_net.mlp(features)              # [N, E]
        gating_weights = gating_net.router(gating_logits)     # [N, E]

        # Check for NaN
        if torch.isnan(gating_weights).any():
            print("WARNING: Gating produces NaN! Falling back to uniform weights")
            N, E = expert_logits.shape[0], expert_logits.shape[1]
            gating_weights = torch.ones(N, E, device=device) / E

        # Verify gating weights sum to 1
        weight_sum = gating_weights.sum(dim=1)
        if not torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5):
            print(
                f"WARNING: Gating weights don't sum to 1! Mean sum: {weight_sum.mean().item():.6f}"
            )

        # Mixture posterior: η̃(x) = Σ_e w_e · p^(e)(y|x)
        mixture_posterior = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(
            dim=1
        )  # [N, C]

        # Verify mixture is valid probability distribution
        mixture_sum = mixture_posterior.sum(dim=1)
        if not torch.allclose(mixture_sum, torch.ones_like(mixture_sum), atol=1e-5):
            print(
                f"WARNING: Mixture posterior doesn't sum to 1! Mean sum: {mixture_sum.mean().item():.6f}"
            )

        return mixture_posterior


def ensure_dirs():
    Path(CFG.results_dir).mkdir(parents=True, exist_ok=True)


# ================================
# Group construction per paper
# ================================


def build_class_to_group() -> torch.Tensor:
    counts_path = Path(CFG.splits_dir) / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(CFG.num_classes)]
    counts = np.array(class_counts)
    tail_mask = counts <= CFG.tail_threshold
    class_to_group = np.zeros(CFG.num_classes, dtype=np.int64)
    class_to_group[tail_mask] = 1  # 0=head, 1=tail
    print(
        f"Groups: head={(class_to_group == 0).sum()}, tail={(class_to_group == 1).sum()}"
    )
    return torch.tensor(class_to_group, dtype=torch.long, device=DEVICE)


# ================================
# Plug-in model (Theorem 1)
# ================================
class BalancedLtRPlugin(nn.Module):
    def __init__(self, class_to_group: torch.Tensor):
        super().__init__()
        self.class_to_group = class_to_group  # [C]
        num_groups = int(class_to_group.max().item() + 1)
        self.register_buffer("alpha_group", torch.ones(num_groups))  # α[g]
        self.register_buffer("mu_group", torch.zeros(num_groups))  # μ[g]
        self.register_buffer("cost", torch.tensor(0.0))  # c

    def set_params(self, alpha_g: torch.Tensor, mu_g: torch.Tensor, cost: float):
        self.alpha_group = alpha_g.to(self.alpha_group.device)
        self.mu_group = mu_g.to(self.mu_group.device)
        self.cost = torch.tensor(float(cost), device=self.cost.device)

    def _alpha_class(self) -> torch.Tensor:
        return self.alpha_group[self.class_to_group]

    def _mu_class(self) -> torch.Tensor:
        return self.mu_group[self.class_to_group]

    def _alpha_hat_class(self) -> torch.Tensor:
        # α̂_k = α_k · β_k ; for balanced β_k = 1/K ⇒ α̂ = α / K
        K = float(self.alpha_group.numel())
        alpha_hat_group = self.alpha_group / max(K, 1.0)
        return alpha_hat_group[self.class_to_group]

    @torch.no_grad()
    def predict(self, posterior: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        alpha_hat = self._alpha_hat_class().clamp(min=eps)
        reweighted = posterior / alpha_hat.unsqueeze(0)
        return reweighted.argmax(dim=-1)

    @torch.no_grad()
    def reject(
        self, posterior: torch.Tensor, cost: Optional[float] = None
    ) -> torch.Tensor:
        eps = 1e-12
        alpha_hat = self._alpha_hat_class().clamp(min=eps)
        mu = self._mu_class()
        inv_alpha_hat = 1.0 / alpha_hat
        max_reweighted = (posterior * inv_alpha_hat.unsqueeze(0)).max(dim=-1)[0]
        threshold = ((inv_alpha_hat - mu).unsqueeze(0) * posterior).sum(dim=-1)
        c = self.cost.item() if cost is None else float(cost)
        return max_reweighted < (threshold - c)


# ================================
# Metrics
# ================================
@torch.no_grad()
def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    reject: torch.Tensor,
    class_to_group: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    accept = ~reject
    if accept.sum() == 0:
        num_groups = int(class_to_group.max().item() + 1)
        return {
            "selective_error": 1.0,
            "coverage": 0.0,
            "group_errors": [1.0] * num_groups,
            "balanced_error": 1.0,
            "worst_group_error": 1.0,
        }
    preds_a = preds[accept]
    labels_a = labels[accept]
    errors = (preds_a != labels_a).float()
    selective_error = errors.mean().item()
    coverage = accept.float().mean().item()
    groups = class_to_group[labels_a]
    num_groups = int(class_to_group.max().item() + 1)

    group_errors = []

    if class_weights is not None:
        device = labels.device
        class_weights = class_weights.to(device)

        for g in range(num_groups):
            mask = groups == g

            if mask.sum() == 0:
                group_errors.append(1.0)
            else:
                y_g = labels_a[mask]
                preds_g = preds_a[mask]

                sample_weights = class_weights[y_g]
                errors_in_group = (preds_g != y_g).float()

                weighted_errors = (sample_weights * errors_in_group).sum().item()
                total_weight = sample_weights.sum().item()

                if total_weight > 0:
                    group_errors.append(weighted_errors / total_weight)
                else:
                    group_errors.append(1.0)
    else:
        for g in range(num_groups):
            mask = groups == g

            if mask.sum() == 0:
                group_errors.append(1.0)
            else:
                num_errors_in_group = errors[mask].sum().item()
                num_accepted_in_group = mask.sum().item()
                conditional_error = num_errors_in_group / num_accepted_in_group
                group_errors.append(conditional_error)

    balanced_error = float(np.mean(group_errors))
    worst_group_error = float(np.max(group_errors))
    return {
        "selective_error": selective_error,
        "coverage": coverage,
        "group_errors": group_errors,
        "balanced_error": balanced_error,
        "worst_group_error": worst_group_error,
    }


# ================================
# Algorithm 1 (Power-iteration)
# ================================
@torch.no_grad()
def initialize_alpha(labels: torch.Tensor, class_to_group: torch.Tensor) -> np.ndarray:
    K = int(class_to_group.max().item() + 1)
    alpha = np.zeros(K, dtype=np.float64)
    N = len(labels)
    for g in range(K):
        group_mask = class_to_group[labels] == g
        prop = group_mask.sum().float().item() / max(N, 1)
        alpha[g] = float(K * prop)
    return alpha


@torch.no_grad()
def update_alpha_from_coverage(
    reject: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> np.ndarray:
    K = int(class_to_group.max().item() + 1)
    alpha = np.zeros(K, dtype=np.float64)
    accept = ~reject
    N = len(labels)
    if accept.sum() == 0:
        return np.ones(K, dtype=np.float64) * 0.5

    if class_weights is not None:
        cw = class_weights.to(labels.device)
        sample_w = cw[labels]
        total_weight = sample_w.sum().item()
        total_weight = max(total_weight, 1e-12)
        for g in range(K):
            in_group = class_to_group[labels] == g
            accepted_in_group = accept & in_group
            w_acc_g = sample_w[accepted_in_group].sum().item()
            cov_g = float(np.clip(w_acc_g / total_weight, 1e-12, 1.0))
            alpha[g] = float(K * cov_g)
        return alpha

    for g in range(K):
        in_group = class_to_group[labels] == g
        accepted_in_group = accept & in_group
        empirical_cov = accepted_in_group.sum().float().item() / max(N, 1)
        alpha[g] = float(K * np.clip(empirical_cov, 1e-6, 1.0))
    return alpha


@torch.no_grad()
def power_iter_search(
    plugin: BalancedLtRPlugin,
    posterior: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    mu: np.ndarray,
    cost: float,
    num_iters: int,
    damping: float,
    class_weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
    target_rejection: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    alpha = initialize_alpha(labels, class_to_group)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    K = int(class_to_group.max().item() + 1)

    for it in range(num_iters):
        alpha_hat = alpha.astype(np.float32)
        alpha_t = torch.tensor(alpha_hat, dtype=torch.float32, device=DEVICE)
        c_it = cost
        if target_rejection is not None:
            c_it = compute_cost_for_target_rejection(
                posterior, class_to_group, alpha, mu, target_rejection
            )
        plugin.set_params(alpha_t, mu_t, c_it)
        preds = plugin.predict(posterior)
        rej = plugin.reject(posterior)
        alpha_new = update_alpha_from_coverage(
            rej, labels, class_to_group, class_weights=class_weights
        )
        if damping > 0.0:
            alpha = (1.0 - damping) * alpha + damping * alpha_new
        else:
            alpha = alpha_new
        if verbose and (it % 10 == 0 or it == num_iters - 1):
            m = compute_metrics(preds, labels, rej, class_to_group, class_weights)
            print(
                f"   [PI] iter={it + 1} cov={m['coverage']:.3f} bal={m['balanced_error']:.4f}"
            )
        if np.max(np.abs(alpha_new - alpha)) < 1e-4:
            break

    alpha_hat = alpha.astype(np.float32)
    alpha_t = torch.tensor(alpha_hat, dtype=torch.float32, device=DEVICE)
    c_fin = cost
    if target_rejection is not None:
        c_fin = compute_cost_for_target_rejection(
            posterior, class_to_group, alpha, mu, target_rejection
        )
    plugin.set_params(alpha_t, mu_t, c_fin)
    preds = plugin.predict(posterior)
    rej = plugin.reject(posterior)
    metrics = compute_metrics(preds, labels, rej, class_to_group, class_weights)
    return alpha, metrics


@torch.no_grad()
def compute_cost_for_target_rejection(
    posterior: torch.Tensor,
    class_to_group: torch.Tensor,
    alpha: np.ndarray,
    mu: np.ndarray,
    target_rejection: float,
) -> float:
    """Compute cost c to achieve a target rejection rate r."""
    eps = 1e-12
    K = int(class_to_group.max().item() + 1)
    C = int(class_to_group.numel())
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    if alpha_t.numel() == K:
        alpha_group = alpha_t
        mu_group = mu_t
        alpha_hat_group = alpha_group / max(float(K), 1.0)
        alpha_t = alpha_hat_group[class_to_group]
        mu_t = mu_group[class_to_group]
    inv_alpha_hat = 1.0 / alpha_t.clamp(min=eps)
    max_rew = (posterior * inv_alpha_hat.unsqueeze(0)).max(dim=-1)[0]
    thresh_base = ((inv_alpha_hat - mu_t).unsqueeze(0) * posterior).sum(dim=-1)
    t = thresh_base - max_rew
    t_sorted = torch.sort(t)[0]
    q = max(0.0, min(1.0, 1.0 - float(target_rejection)))
    idx = int(round(q * (len(t_sorted) - 1)))
    return float(t_sorted[idx].item())


# ================================
# RC curve (balanced error) and AURC
# ================================
@torch.no_grad()
def compute_rc_curve(
    plugin: BalancedLtRPlugin,
    posterior: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    alpha: np.ndarray,
    mu: np.ndarray,
    cost_grid: List[float],
) -> Dict[str, np.ndarray]:
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    rejection_rates, balanced_errors = [], []
    for c in cost_grid:
        plugin.set_params(alpha_t, mu_t, c)
        preds = plugin.predict(posterior)
        rej = plugin.reject(posterior)
        m = compute_metrics(preds, labels, rej, class_to_group)
        rejection_rates.append(1.0 - m["coverage"])
        balanced_errors.append(m["balanced_error"])
    r = np.array(rejection_rates)
    e = np.array(balanced_errors)
    idx = np.argsort(r)
    r, e = r[idx], e[idx]
    aurc = float(np.trapz(e, r))
    return {"rejection_rates": r, "balanced_errors": e, "aurc": aurc}


# ================================
# Plotting
# ================================


def plot_rc_dual(
    r: np.ndarray,
    e_bal: np.ndarray,
    e_wst: np.ndarray,
    aurc_bal: float,
    aurc_wst: float,
    out_path: Path,
):
    plt.figure(figsize=(7, 5))
    plt.plot(r, e_bal, "o-", color="green", label=f"Balanced (AURC={aurc_bal:.4f})")
    plt.plot(
        r, e_wst, "s-", color="royalblue", label=f"Worst-group (AURC={aurc_wst:.4f})"
    )
    plt.xlabel("Proportion of Rejections")
    plt.ylabel("Error")
    plt.title("Balanced and Worst-group Error vs Rejection Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    ymax = 0.0
    if e_bal.size:
        ymax = max(ymax, float(e_bal.max()))
    if e_wst.size:
        ymax = max(ymax, float(e_wst.max()))
    plt.ylim([0, min(1.05, ymax * 1.1 if ymax > 0 else 1.0)])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ================================
# Main
# ================================


def main():
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    ensure_dirs()

    print("Loading gating network...")
    gating = load_gating_network(DEVICE)

    print("Loading S1 (tunev) and S2 (val) for selection/evaluation...")
    expert_logits_tunev = load_expert_logits(CFG.expert_names, "tunev", DEVICE)
    labels_tunev = load_labels("tunev", DEVICE)

    expert_logits_val = load_expert_logits(CFG.expert_names, "val", DEVICE)
    labels_val = load_labels("val", DEVICE)

    print("Computing mixture posteriors using gating network...")
    posterior_tunev = compute_mixture_posterior(expert_logits_tunev, gating, DEVICE)
    posterior_val = compute_mixture_posterior(expert_logits_val, gating, DEVICE)

    print("Building class-to-group mapping (tail <= 20)...")
    class_to_group = build_class_to_group()

    print("Loading class weights for importance weighting...")
    class_weights = load_class_weights(DEVICE)

    # Check baseline balanced error on test set
    expert_logits_test = load_expert_logits(CFG.expert_names, "test", DEVICE)
    labels_test = load_labels("test", DEVICE)
    posterior_test = compute_mixture_posterior(expert_logits_test, gating, DEVICE)

    # Compute baseline balanced error (no rejection) with importance weighting
    mix_pred_test = posterior_test.argmax(dim=-1)
    groups_test = class_to_group[labels_test]
    num_groups = int(class_to_group.max().item() + 1)

    dummy_reject = torch.zeros(len(labels_test), dtype=torch.bool, device=DEVICE)
    baseline_metrics = compute_metrics(
        mix_pred_test, labels_test, dummy_reject, class_to_group, class_weights
    )

    baseline_balanced_error = baseline_metrics["balanced_error"]
    print(f"Baseline Gating balanced error (TEST) = {baseline_balanced_error:.4f}")
    print(f"Baseline Gating group errors = {baseline_metrics['group_errors']}")
    print(
        f"Baseline Gating overall accuracy (TEST) = {(mix_pred_test == labels_test).float().mean().item():.4f}"
    )

    print("Creating plug-in model...")
    plugin = BalancedLtRPlugin(class_to_group).to(DEVICE)

    # Target rejection grid (paper-style points)
    results_per_cost: List[Dict] = []
    targets = list(CFG.target_rejections)
    for i, target_rej in enumerate(targets):
        print(f"\n=== Target {i + 1}/{len(targets)}: rejection={target_rej:.1f} ===")

        # Follow paper Algorithm 1 - optimize on tunev, select on val
        print("   Step 1: Optimizing (alpha, mu) on tunev for each mu...")
        candidates = []
        for lam in CFG.mu_lambda_grid:
            mu = np.array([0.0, float(lam)], dtype=np.float64)
            alpha_found, _ = power_iter_search(
                plugin,
                posterior_tunev,
                labels_tunev,
                class_to_group,
                mu=mu,
                cost=0.0,
                num_iters=CFG.power_iter_iters,
                damping=CFG.power_iter_damping,
                class_weights=class_weights,
                verbose=False,
                target_rejection=target_rej,
            )
            candidates.append((alpha_found, mu))
            print(f"     mu={lam:5.1f}: alpha={alpha_found}")

        # Select best mu based on val performance
        print("   Step 2: Selecting best mu based on val performance...")
        best = {
            "objective": float("inf"),
            "alpha": None,
            "mu": None,
            "val_metrics": None,
        }

        for alpha, mu in candidates:
            K = int(class_to_group.max().item() + 1)
            alpha_eval = alpha.astype(np.float32)

            cost_val = compute_cost_for_target_rejection(
                posterior_val, class_to_group, alpha, mu, target_rej
            )

            plugin.set_params(
                torch.tensor(alpha_eval, dtype=torch.float32, device=DEVICE),
                torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                float(cost_val),
            )
            preds_val = plugin.predict(posterior_val)
            rej_val = plugin.reject(posterior_val)
            m_val = compute_metrics(
                preds_val, labels_val, rej_val, class_to_group, class_weights
            )

            print(
                f"     mu={mu[1]:5.1f}: val_bal={m_val['balanced_error']:.4f} val_cov={m_val['coverage']:.3f}"
            )

            if m_val["balanced_error"] < best["objective"]:
                best = {
                    "objective": m_val["balanced_error"],
                    "alpha": alpha,
                    "mu": mu,
                    "val_metrics": m_val,
                }

        # Local refinement around best mu
        print("   Step 3: Local refinement around best mu...")
        best_lam = float(best["mu"][1])
        refine_step = 2.0
        for refine_iter in range(4):
            tried = []
            for lam in (best_lam - refine_step, best_lam + refine_step):
                mu = np.array([0.0, float(lam)], dtype=np.float64)
                alpha_found, _ = power_iter_search(
                    plugin,
                    posterior_tunev,
                    labels_tunev,
                    class_to_group,
                    mu=mu,
                    cost=0.0,
                    num_iters=CFG.power_iter_iters,
                    damping=CFG.power_iter_damping,
                    class_weights=class_weights,
                    verbose=False,
                    target_rejection=target_rej,
                )

                alpha_eval = alpha_found.astype(np.float32)
                cost_val = compute_cost_for_target_rejection(
                    posterior_val, class_to_group, alpha_found, mu, target_rej
                )
                plugin.set_params(
                    torch.tensor(alpha_eval, dtype=torch.float32, device=DEVICE),
                    torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                    float(cost_val),
                )
                preds_val = plugin.predict(posterior_val)
                rej_val = plugin.reject(posterior_val)
                m_val = compute_metrics(
                    preds_val, labels_val, rej_val, class_to_group, class_weights
                )

                tried.append((lam, m_val["balanced_error"], alpha_found, mu, m_val))

            lam_better, obj_better, alpha_better, mu_better, metr_better = min(
                tried, key=lambda x: x[1]
            )
            if obj_better < best["objective"]:
                best = {
                    "objective": obj_better,
                    "alpha": alpha_better,
                    "mu": mu_better,
                    "val_metrics": metr_better,
                }
                best_lam = lam_better
                print(
                    f"     Refine {refine_iter + 1}: mu={lam_better:.2f} val_bal={obj_better:.4f}"
                )
            refine_step *= 0.5

        print(
            f"   Final selection: mu={best['mu'][1]:.2f} val_bal={best['val_metrics']['balanced_error']:.4f} val_cov={best['val_metrics']['coverage']:.3f}"
        )
        print(f"   Best alpha: {best['alpha']}")
        K_groups = int(class_to_group.max().item() + 1)
        print(
            f"   Alpha sum: {np.sum(best['alpha']):.4f} (should be ≈ {K_groups * best['val_metrics']['coverage']:.4f})"
        )
        print(f"   Alpha_hat (=alpha/K): {best['alpha'] / float(K_groups)}")

        # Evaluate best (alpha, mu, c) on val and test
        alpha_best = np.array(best["alpha"], dtype=np.float64)
        mu_best = np.array(best["mu"], dtype=np.float64)

        m_val = best["val_metrics"]
        if len(m_val["group_errors"]) >= 2:
            head_err_v = float(m_val["group_errors"][0])
            tail_err_v = float(m_val["group_errors"][1])
            gap_v = tail_err_v - head_err_v
        else:
            head_err_v = tail_err_v = gap_v = float("nan")

        cost_test = compute_cost_for_target_rejection(
            posterior_test, class_to_group, alpha_best, mu_best, target_rej
        )

        K = int(class_to_group.max().item() + 1)
        alpha_eval = alpha_best.astype(np.float32)
        plugin.set_params(
            torch.tensor(alpha_eval, dtype=torch.float32, device=DEVICE),
            torch.tensor(mu_best, dtype=torch.float32, device=DEVICE),
            float(cost_test),
        )
        preds_test = plugin.predict(posterior_test)
        rej_test = plugin.reject(posterior_test)
        m_test = compute_metrics(
            preds_test, labels_test, rej_test, class_to_group, class_weights
        )

        print(
            f"   Target={target_rej:.1f}  VAL: bal={m_val['balanced_error']:.4f} cov={m_val['coverage']:.3f}"
        )
        print(
            f"   Target={target_rej:.1f}  TEST: bal={m_test['balanced_error']:.4f} cov={m_test['coverage']:.3f}"
        )

        wge_v = (
            float(m_val["worst_group_error"])
            if "worst_group_error" in m_val
            else float("nan")
        )
        if len(m_test["group_errors"]) >= 2:
            head_err_t = float(m_test["group_errors"][0])
            tail_err_t = float(m_test["group_errors"][1])
            gap_t = tail_err_t - head_err_t
        else:
            head_err_t = tail_err_t = gap_t = float("nan")
        wge_t = (
            float(m_test["worst_group_error"])
            if "worst_group_error" in m_test
            else float("nan")
        )
        print(
            f"      VAL: worst={wge_v:.4f} | head={head_err_v:.4f} | tail={tail_err_v:.4f} | tail-head={gap_v:.4f}"
        )
        print(
            f"      TEST: worst={wge_t:.4f} | head={head_err_t:.4f} | tail={tail_err_t:.4f} | tail-head={gap_t:.4f}"
        )

        if abs(target_rej - 0.0) < 1e-6:
            print(f"   alpha (group) learned = {alpha_best}")
            mix_pred_test = posterior_test.argmax(dim=-1)
            mix_acc_test = (mix_pred_test == labels_test).float().mean().item()
            plugin_acc_test = 1.0 - m_test["balanced_error"]
            print(f"   Baseline Gating acc (TEST) = {mix_acc_test:.4f}")
            print(f"   Plugin balanced complement (1-bErr) ~ {plugin_acc_test:.4f}")

        results_per_cost.append(
            {
                "target_rejection": float(target_rej),
                "cost_val": float(
                    compute_cost_for_target_rejection(
                        posterior_val, class_to_group, alpha_best, mu_best, target_rej
                    )
                ),
                "cost_test": float(cost_test),
                "alpha": alpha_best.tolist(),
                "mu": mu_best.tolist(),
                "selection_method": "val_based",
                "val_metrics": {
                    "balanced_error": float(m_val["balanced_error"]),
                    "worst_group_error": float(m_val["worst_group_error"]),
                    "coverage": float(m_val["coverage"]),
                    "rejection_rate": float(1.0 - m_val["coverage"]),
                    "group_errors": [float(x) for x in m_val["group_errors"]],
                },
                "test_metrics": {
                    "balanced_error": float(m_test["balanced_error"]),
                    "worst_group_error": float(m_test["worst_group_error"]),
                    "coverage": float(m_test["coverage"]),
                    "rejection_rate": float(1.0 - m_test["coverage"]),
                    "group_errors": [float(x) for x in m_test["group_errors"]],
                },
            }
        )

    # Build unified RC curve (balanced) from target points
    r_val = np.array([1.0 - r["val_metrics"]["coverage"] for r in results_per_cost])
    e_val = np.array([r["val_metrics"]["balanced_error"] for r in results_per_cost])
    w_val = np.array([r["val_metrics"]["worst_group_error"] for r in results_per_cost])
    gap_val = np.array(
        [
            r["val_metrics"]["group_errors"][1] - r["val_metrics"]["group_errors"][0]
            for r in results_per_cost
        ]
    )
    r_test = np.array([1.0 - r["test_metrics"]["coverage"] for r in results_per_cost])
    e_test = np.array([r["test_metrics"]["balanced_error"] for r in results_per_cost])
    w_test = np.array(
        [r["test_metrics"]["worst_group_error"] for r in results_per_cost]
    )
    gap_test = np.array(
        [
            r["test_metrics"]["group_errors"][1] - r["test_metrics"]["group_errors"][0]
            for r in results_per_cost
        ]
    )

    idx_v = np.argsort(r_val)
    r_val, e_val, w_val, gap_val = (
        r_val[idx_v],
        e_val[idx_v],
        w_val[idx_v],
        gap_val[idx_v],
    )
    idx_t = np.argsort(r_test)
    r_test, e_test, w_test, gap_test = (
        r_test[idx_t],
        e_test[idx_t],
        w_test[idx_t],
        gap_test[idx_t],
    )

    aurc_val_bal = (
        float(np.trapz(e_val, r_val))
        if r_val.size > 1
        else float(e_val.mean() if e_val.size else 0.0)
    )
    aurc_test_bal = (
        float(np.trapz(e_test, r_test))
        if r_test.size > 1
        else float(e_test.mean() if e_test.size else 0.0)
    )
    aurc_val_wst = (
        float(np.trapz(w_val, r_val))
        if r_val.size > 1
        else float(w_val.mean() if w_val.size else 0.0)
    )
    aurc_test_wst = (
        float(np.trapz(w_test, r_test))
        if r_test.size > 1
        else float(w_test.mean() if w_test.size else 0.0)
    )

    # Practical AURC over coverage >= 0.2 → rejection <= 0.8
    if r_val.size > 1:
        mask_val_08 = r_val <= 0.8
        aurc_val_bal_08 = (
            float(np.trapz(e_val[mask_val_08], r_val[mask_val_08]))
            if mask_val_08.sum() > 1
            else float(e_val[mask_val_08].mean() if mask_val_08.any() else aurc_val_bal)
        )
        aurc_val_wst_08 = (
            float(np.trapz(w_val[mask_val_08], r_val[mask_val_08]))
            if mask_val_08.sum() > 1
            else float(w_val[mask_val_08].mean() if mask_val_08.any() else aurc_val_wst)
        )
    else:
        aurc_val_bal_08 = aurc_val_bal
        aurc_val_wst_08 = aurc_val_wst

    if r_test.size > 1:
        mask_test_08 = r_test <= 0.8
        aurc_test_bal_08 = (
            float(np.trapz(e_test[mask_test_08], r_test[mask_test_08]))
            if mask_test_08.sum() > 1
            else float(
                e_test[mask_test_08].mean() if mask_test_08.any() else aurc_test_bal
            )
        )
        aurc_test_wst_08 = (
            float(np.trapz(w_test[mask_test_08], r_test[mask_test_08]))
            if mask_test_08.sum() > 1
            else float(
                w_test[mask_test_08].mean() if mask_test_08.any() else aurc_test_wst
            )
        )
    else:
        aurc_test_bal_08 = aurc_test_bal
        aurc_test_wst_08 = aurc_test_wst

    save_dict = {
        "objectives": ["balanced", "worst_group"],
        "description": "CORRECTED: Targeted rejection grid (0.0..0.8) with val-based hyperparameter selection per paper Algorithm 1. Uses 3 experts with gating network.",
        "method": "plug-in_balanced_val_selection_gating",
        "hyperparameter_selection": "val_based",
        "algorithm": "Algorithm 1 from paper - optimize (α,μ) on tunev, select μ on val",
        "experts": CFG.expert_names,
        "results_per_cost": results_per_cost,
        "rc_curve": {
            "val": {
                "rejection_rates": r_val.tolist(),
                "balanced_errors": e_val.tolist(),
                "worst_group_errors": w_val.tolist(),
                "tail_minus_head": gap_val.tolist(),
                "aurc_balanced": aurc_val_bal,
                "aurc_worst_group": aurc_val_wst,
                "aurc_balanced_coverage_ge_0_2": aurc_val_bal_08,
                "aurc_worst_group_coverage_ge_0_2": aurc_val_wst_08,
            },
            "test": {
                "rejection_rates": r_test.tolist(),
                "balanced_errors": e_test.tolist(),
                "worst_group_errors": w_test.tolist(),
                "tail_minus_head": gap_test.tolist(),
                "aurc_balanced": aurc_test_bal,
                "aurc_worst_group": aurc_test_wst,
                "aurc_balanced_coverage_ge_0_2": aurc_test_bal_08,
                "aurc_worst_group_coverage_ge_0_2": aurc_test_wst_08,
            },
        },
    }

    out_json = Path(CFG.results_dir) / "ltr_plugin_gating_balanced.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2)
    print(f"Saved results to: {out_json}")

    # Print AURCs
    print(f"Val AURC - Balanced: {aurc_val_bal:.4f} | Worst-group: {aurc_val_wst:.4f}")
    print(
        f"Test AURC - Balanced: {aurc_test_bal:.4f} | Worst-group: {aurc_test_wst:.4f}"
    )
    print(
        f"Val AURC (coverage>=0.2) - Balanced: {aurc_val_bal_08:.4f} | Worst-group: {aurc_val_wst_08:.4f}"
    )
    print(
        f"Test AURC (coverage>=0.2) - Balanced: {aurc_test_bal_08:.4f} | Worst-group: {aurc_test_wst_08:.4f}"
    )

    # Plot test RC curves (both metrics)
    plot_path = Path(CFG.results_dir) / "ltr_rc_curves_balanced_gating_test.png"
    plot_rc_dual(r_test, e_test, w_test, aurc_test_bal, aurc_test_wst, plot_path)
    print(f"Saved combined plot to: {plot_path}")

    # Plot Tail - Head gap curve
    gap_plot_path = Path(CFG.results_dir) / "ltr_tail_minus_head_gating_test.png"
    plt.figure(figsize=(7, 5))
    plt.plot(r_test, gap_test, "d-", color="crimson", label="Tail - Head error")
    plt.xlabel("Proportion of Rejections")
    plt.ylabel("Tail Error - Head Error")
    plt.title("Tail-Head Error Gap vs Rejection Rate")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    ymin = float(min(0.0, gap_test.min() if gap_test.size else 0.0))
    ymax = float(max(0.0, gap_test.max() if gap_test.size else 0.0))
    pad = 0.05 * (ymax - ymin + 1e-8)
    plt.ylim([ymin - pad, ymax + pad])
    plt.tight_layout()
    plt.savefig(gap_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved gap plot to: {gap_plot_path}")


if __name__ == "__main__":
    import sys
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Balanced L2R Plugin with Gating")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100_lt_if100",
        choices=["cifar100_lt_if100", "inaturalist2018"],
        help="Dataset name"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. If provided, all output will be saved to this file"
    )
    args = parser.parse_args()
    
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
        # Setup config based on dataset
        CFG = setup_config(args.dataset)
        
        print(f"✓ Using dataset: {args.dataset}")
        print(f"  Classes: {CFG.num_classes}")
        print(f"  Experts: {CFG.expert_names}")
        
        main()
    finally:
        # Restore stdout and close log file
        if log_file_handle is not None:
            sys.stdout = original_stdout
            log_file_handle.close()
            print(f"\n[Log saved to: {args.log_file}]")
