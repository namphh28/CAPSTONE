#!/usr/bin/env python3
"""
Worst-group Plug-in (CE-only) per "Learning to Reject Meets Long-Tail Learning"
===============================================================================

- Implements Algorithm 2 (Worst-group Plug-in) with inner Algorithm 1 (CS plug-in)
- Reuses CE logits; uses tunev (S1) for optimization and val (S2) for exponentiated-gradient updates
- Evaluates on test; reports RC curves and AURC for worst-group (primary) and balanced (secondary)

Inputs:
- Splits: ./data/cifar100_lt_if100_splits_fixed/
- Logits: ./outputs/logits/cifar100_lt_if100/ce_baseline/{split}_logits.pt

Outputs:
- results/ltr_plugin/cifar100_lt_if100/ltr_plugin_ce_only_worst.json
- results/ltr_plugin/cifar100_lt_if100/ltr_rc_curves_balanced_worst_ce_only_test.png
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


@dataclass
class Config:
    dataset_name: str = "cifar100_lt_if100"
    splits_dir: str = "./data/cifar100_lt_if100_splits_fixed"
    logits_dir: str = "./outputs/logits/cifar100_lt_if100/ce_baseline"
    results_dir: str = "./results/ltr_plugin/cifar100_lt_if100"

    num_classes: int = 100
    num_groups: int = 2
    tail_threshold: int = 20

    # Inner CS plug-in (Algorithm 1) grid over single λ = μ_tail − μ_head
    mu_lambda_grid: List[float] = (1.0, 6.0, 11.0)
    power_iter_iters: int = 10
    power_iter_damping: float = 0.5

    # Algorithm 2 (Worst): EG iterations and step-size
    eg_iters: int = 25
    eg_step: float = 1.0

    # Target rejection grid
    target_rejections: List[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

    seed: int = 42


CFG = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dirs():
    Path(CFG.results_dir).mkdir(parents=True, exist_ok=True)


def load_logits(split: str, device: str = DEVICE) -> torch.Tensor:
    path = Path(CFG.logits_dir) / f"{split}_logits.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing logits: {path}")
    return torch.load(path, map_location=device).float()


def load_labels(split: str, device: str = DEVICE) -> torch.Tensor:
    cand = Path(CFG.logits_dir) / f"{split}_targets.pt"
    if cand.exists():
        t = torch.load(cand, map_location=device)
        if isinstance(t, torch.Tensor):
            return t.to(device=device, dtype=torch.long)
    import torchvision
    indices_file = Path(CFG.splits_dir) / f"{split}_indices.json"
    with open(indices_file, "r", encoding="utf-8") as f:
        indices = json.load(f)
    is_train = split in ("expert", "gating", "train")
    ds = torchvision.datasets.CIFAR100(root="./data", train=is_train, download=False)
    return torch.tensor([ds.targets[i] for i in indices], dtype=torch.long, device=device)


def load_class_weights(device: str = DEVICE) -> torch.Tensor:
    counts_path = Path(CFG.splits_dir) / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(CFG.num_classes)]
    counts = np.array(class_counts, dtype=np.float64)
    total = counts.sum()
    train_probs = counts / max(total, 1e-12)
    weights = train_probs * CFG.num_classes  # test assumed balanced
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_class_to_group() -> torch.Tensor:
    counts_path = Path(CFG.splits_dir) / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(CFG.num_classes)]
    counts = np.array(class_counts)
    tail_mask = counts <= CFG.tail_threshold
    class_to_group = np.zeros(CFG.num_classes, dtype=np.int64)
    class_to_group[tail_mask] = 1
    return torch.tensor(class_to_group, dtype=torch.long, device=DEVICE)


class GeneralizedLtRPlugin(nn.Module):
    """Implements Theorem 12 decision with α, μ and group weights β.

    Reject rule: max_y p_y / (α̂_[y]) < Σ_y (1/α̂_[y] − μ_[y]) p_y − c,
    with α̂_k = α_k · β_k.
    """

    def __init__(self, class_to_group: torch.Tensor):
        super().__init__()
        self.class_to_group = class_to_group
        num_groups = int(class_to_group.max().item() + 1)
        self.register_buffer("alpha_group", torch.ones(num_groups))
        self.register_buffer("mu_group", torch.zeros(num_groups))
        self.register_buffer("beta_group", torch.ones(num_groups) / float(max(num_groups, 1)))
        self.register_buffer("cost", torch.tensor(0.0))

    def set_params(
        self,
        alpha_g: torch.Tensor,
        mu_g: torch.Tensor,
        beta_g: torch.Tensor,
        cost: float,
    ):
        self.alpha_group = alpha_g.to(self.alpha_group.device)
        self.mu_group = mu_g.to(self.mu_group.device)
        self.beta_group = beta_g.to(self.beta_group.device)
        self.cost = torch.tensor(float(cost), device=self.cost.device)

    def _u_class(self) -> torch.Tensor:
        # u = β / α per group → expand to class level
        eps = 1e-12
        u_group = (self.beta_group / self.alpha_group.clamp(min=eps))
        return u_group[self.class_to_group]

    def _mu_class(self) -> torch.Tensor:
        return self.mu_group[self.class_to_group]

    @torch.no_grad()
    def predict(self, posterior: torch.Tensor) -> torch.Tensor:
        u = self._u_class().unsqueeze(0)
        return (posterior * u).argmax(dim=-1)

    @torch.no_grad()
    def reject(self, posterior: torch.Tensor, cost: Optional[float] = None) -> torch.Tensor:
        u = self._u_class().unsqueeze(0)
        mu = self._mu_class().unsqueeze(0)
        max_reweighted = (posterior * u).max(dim=-1)[0]
        threshold = ((u - mu) * posterior).sum(dim=-1)
        c = self.cost.item() if cost is None else float(cost)
        return max_reweighted < (threshold - c)


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
        K = int(class_to_group.max().item() + 1)
        return {
            "selective_error": 1.0,
            "coverage": 0.0,
            "group_errors": [1.0] * K,
            "balanced_error": 1.0,
            "worst_group_error": 1.0,
        }
    preds_a = preds[accept]
    labels_a = labels[accept]
    groups_a = class_to_group[labels_a]
    errors = (preds_a != labels_a).float()
    K = int(class_to_group.max().item() + 1)
    group_errors = []

    # Coverage to report on RC x-axis should be UNWEIGHTED (paper: proportion of rejections)
    cov_unw = float(accept.float().mean().item())
    cov_w = None
    if class_weights is not None:
        cw_all = class_weights[labels]
        total_w = float(cw_all.sum().item())
        cov_w = float(cw_all[accept].sum().item() / max(total_w, 1e-12))

    for g in range(K):
        mask = groups_a == g
        if mask.sum() == 0:
            group_errors.append(1.0)
            continue
        if class_weights is None:
            group_errors.append(float(errors[mask].mean().item()))
        else:
            cw = class_weights[labels_a]
            num = float((cw[mask] * errors[mask]).sum().item())
            den = float(cw[mask].sum().item())
            group_errors.append(num / max(den, 1e-12))

    balanced_error = float(np.mean(group_errors))
    worst_group_error = float(np.max(group_errors))
    return {
        "selective_error": float(errors.mean().item()),
        "coverage": cov_unw,
        "coverage_weighted": cov_w if cov_w is not None else cov_unw,
        "group_errors": group_errors,
        "balanced_error": balanced_error,
        "worst_group_error": worst_group_error,
    }


@torch.no_grad()
def initialize_alpha(labels: torch.Tensor, class_to_group: torch.Tensor) -> np.ndarray:
    # α_k initialized as coverage probabilities in (0,1): α_k ≈ P(y ∈ G_k)
    K = int(class_to_group.max().item() + 1)
    alpha = np.zeros(K, dtype=np.float64)
    for g in range(K):
        prop = (class_to_group[labels] == g).float().mean().item()
        alpha[g] = float(max(prop, 1e-12))
    return alpha


@torch.no_grad()
def update_alpha_from_coverage(
    reject: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> np.ndarray:
    # α_k = P(r=0, y ∈ G_k) ∈ (0,1)
    K = int(class_to_group.max().item() + 1)
    alpha = np.zeros(K, dtype=np.float64)
    accept = ~reject
    N = max(1, len(labels))
    if class_weights is None:
        # Unweighted fallback
        for g in range(K):
            in_group = class_to_group[labels] == g
            cov_g = (accept & in_group).float().sum().item() / N
            alpha[g] = float(max(cov_g, 1e-12))
        return alpha

    # Weighted α using train-prior importance weights
    cw_all = class_weights[labels]
    total_w = float(cw_all.sum().item())
    for g in range(K):
        in_group = class_to_group[labels] == g
        cov_w = float(cw_all[accept & in_group].sum().item() / max(total_w, 1e-12))
        alpha[g] = float(max(cov_w, 1e-12))
    return alpha


@torch.no_grad()
def compute_cost_for_target_rejection(
    posterior: torch.Tensor,
    class_to_group: torch.Tensor,
    alpha: np.ndarray,
    mu: np.ndarray,
    beta: np.ndarray,
    target_rejection: float,
) -> float:
    # Use u = β/α; threshold t(x) = Σ (u − μ) p − max_y u_y p_y
    K = int(class_to_group.max().item() + 1)
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    beta_t = torch.tensor(beta, dtype=torch.float32, device=DEVICE)
    u_group = (beta_t / alpha_t.clamp(min=1e-12))
    u_class = u_group[class_to_group]
    mu_class = mu_t[class_to_group]
    max_rew = (posterior * u_class.unsqueeze(0)).max(dim=-1)[0]
    thresh_base = ((u_class - mu_class).unsqueeze(0) * posterior).sum(dim=-1)
    t = thresh_base - max_rew
    t_sorted = torch.sort(t)[0]
    q = max(0.0, min(1.0, 1.0 - float(target_rejection)))
    idx = int(round(q * (len(t_sorted) - 1)))
    return float(t_sorted[idx].item())


@torch.no_grad()
def cs_plugin_inner(
    plugin: GeneralizedLtRPlugin,
    posterior_s1: torch.Tensor,
    labels_s1: torch.Tensor,
    posterior_s2: torch.Tensor,
    labels_s2: torch.Tensor,
    class_to_group: torch.Tensor,
    beta: np.ndarray,
    target_rej: float,
    class_weights: Optional[torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    """Algorithm 1 adapted: search λ over grid; use power-iteration on S1; select on S2 by weighted error Σ β_k e_k."""
    K = int(class_to_group.max().item() + 1)
    best = {"obj": float("inf"), "alpha": None, "mu": None, "cost": None, "metrics": None}
    print(f"[CS] target_rej={target_rej:.3f} beta={beta}")
    for lam in CFG.mu_lambda_grid:
        mu = np.array([0.0, float(lam)], dtype=np.float64) if K == 2 else np.zeros(K, dtype=np.float64)
        # power-iteration on S1
        alpha = initialize_alpha(labels_s1, class_to_group)
        for it in range(CFG.power_iter_iters):
            beta_t = torch.tensor(beta, dtype=torch.float32, device=DEVICE)
            plugin.set_params(
                torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
                torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                beta_t,
                0.0,
            )
            c_it = compute_cost_for_target_rejection(
                posterior_s1, class_to_group, alpha, mu, beta, target_rej
            )
            plugin.set_params(
                torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
                torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                beta_t,
                float(c_it),
            )
            rej = plugin.reject(posterior_s1)
            alpha_new = update_alpha_from_coverage(rej, labels_s1, class_to_group, class_weights)
            alpha = (1.0 - CFG.power_iter_damping) * alpha + CFG.power_iter_damping * alpha_new
            if (it % 5) == 0 or it == CFG.power_iter_iters - 1:
                preds_dbg = plugin.predict(posterior_s1)
                m_dbg = compute_metrics(preds_dbg, labels_s1, rej, class_to_group, class_weights)
                print(
                    f"   [CS][lam={lam:.2f}] it={it+1}/{CFG.power_iter_iters} cov={m_dbg['coverage']:.3f} bal={m_dbg['balanced_error']:.4f} wst={m_dbg['worst_group_error']:.4f} alpha_sum={alpha.sum():.4f}"
                )
        # select on S2
        c_s2 = compute_cost_for_target_rejection(
            posterior_s2, class_to_group, alpha, mu, beta, target_rej
        )
        plugin.set_params(
            torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
            torch.tensor(mu, dtype=torch.float32, device=DEVICE),
            torch.tensor(beta, dtype=torch.float32, device=DEVICE),
            float(c_s2),
        )
        preds_s2 = plugin.predict(posterior_s2)
        rej_s2 = plugin.reject(posterior_s2)
        m_s2 = compute_metrics(preds_s2, labels_s2, rej_s2, class_to_group, class_weights)
        # objective Σ β_k e_k
        obj = float(np.sum(np.array(m_s2["group_errors"]) * np.array(beta)))
        print(
            f"   [VAL][lam={lam:.2f}] obj={obj:.4f} cov={m_s2['coverage']:.3f} bal={m_s2['balanced_error']:.4f} wst={m_s2['worst_group_error']:.4f} cost={c_s2:.5f}"
        )
        if obj < best["obj"]:
            best = {"obj": obj, "alpha": alpha.copy(), "mu": mu.copy(), "cost": float(c_s2), "metrics": m_s2}

    # Local refine around best μ (λ)
    if K == 2 and best["mu"] is not None:
        base_lam = float(best["mu"][1])
        step = 2.0
        for it_ref in range(4):
            tried = []
            for lam in (base_lam - step, base_lam + step):
                mu = np.array([0.0, float(lam)], dtype=np.float64)
                # Re-run PI on S1
                alpha = initialize_alpha(labels_s1, class_to_group)
                for it in range(max(8, CFG.power_iter_iters // 2)):
                    plugin.set_params(
                        torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
                        torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                        torch.tensor(beta, dtype=torch.float32, device=DEVICE),
                        0.0,
                    )
                    c_it = compute_cost_for_target_rejection(
                        posterior_s1, class_to_group, alpha, mu, beta, target_rej
                    )
                    plugin.set_params(
                        torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
                        torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                        torch.tensor(beta, dtype=torch.float32, device=DEVICE),
                        float(c_it),
                    )
                    rej = plugin.reject(posterior_s1)
                    alpha_new = update_alpha_from_coverage(rej, labels_s1, class_to_group, class_weights)
                    alpha = (1.0 - CFG.power_iter_damping) * alpha + CFG.power_iter_damping * alpha_new
                # Evaluate on S2
                c_s2 = compute_cost_for_target_rejection(
                    posterior_s2, class_to_group, alpha, mu, beta, target_rej
                )
                plugin.set_params(
                    torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
                    torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                    torch.tensor(beta, dtype=torch.float32, device=DEVICE),
                    float(c_s2),
                )
                preds_s2 = plugin.predict(posterior_s2)
                rej_s2 = plugin.reject(posterior_s2)
                m_s2 = compute_metrics(preds_s2, labels_s2, rej_s2, class_to_group, class_weights)
                obj = float(np.sum(np.array(m_s2["group_errors"]) * np.array(beta)))
                tried.append((lam, obj, alpha.copy(), mu.copy(), float(c_s2), m_s2))
            lam_b, obj_b, alpha_b, mu_b, c_b, m_b = min(tried, key=lambda x: x[1])
            if obj_b < best["obj"]:
                best = {"obj": obj_b, "alpha": alpha_b, "mu": mu_b, "cost": c_b, "metrics": m_b}
                base_lam = lam_b
                print(f"   [REFINE] lam={lam_b:.3f} obj={obj_b:.4f} cov={m_b['coverage']:.3f}")
            step *= 0.5

    return best["alpha"], best["mu"], best["cost"], best["metrics"]


def plot_rc(r: np.ndarray, ew: np.ndarray, eb: np.ndarray, aw: float, ab: float, out_path: Path):
    plt.figure(figsize=(7, 5))
    plt.plot(r, ew, "o-", label=f"Worst-group (AURC={aw:.4f})", color="royalblue")
    plt.plot(r, eb, "s-", label=f"Balanced (AURC={ab:.4f})", color="green")
    plt.xlabel("Proportion of Rejections")
    plt.ylabel("Error")
    plt.title("Worst-group and Balanced Error vs Rejection Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    ensure_dirs()

    logits_s1 = load_logits("tunev", DEVICE)
    labels_s1 = load_labels("tunev", DEVICE)
    logits_s2 = load_logits("val", DEVICE)
    labels_s2 = load_labels("val", DEVICE)
    logits_test = load_logits("test", DEVICE)
    labels_test = load_labels("test", DEVICE)
    class_to_group = build_class_to_group()
    class_weights = load_class_weights(DEVICE)

    post_s1 = F.softmax(logits_s1, dim=-1)
    post_s2 = F.softmax(logits_s2, dim=-1)
    post_test = F.softmax(logits_test, dim=-1)

    plugin = GeneralizedLtRPlugin(class_to_group).to(DEVICE)

    results = []
    for target_rej in CFG.target_rejections:
        # EG over β
        K = int(class_to_group.max().item() + 1)
        beta = np.ones(K, dtype=np.float64) / float(K)
        alpha_best = None
        mu_best = None
        cost_best = None
        for t in range(CFG.eg_iters):
            alpha_t, mu_t, cost_t, metr_s2 = cs_plugin_inner(
                plugin, post_s1, labels_s1, post_s2, labels_s2, class_to_group, beta, float(target_rej), class_weights
            )
            # Update β via EG using group errors on S2
            e = np.array(metr_s2["group_errors"], dtype=np.float64)
            beta = beta * np.exp(CFG.eg_step * e)
            s = max(np.sum(beta), 1e-12)
            beta = beta / s
            alpha_best, mu_best, cost_best = alpha_t, mu_t, cost_t
            print(
                f"[EG] t={t+1}/{CFG.eg_iters} beta={beta} val_bal={metr_s2['balanced_error']:.4f} val_wst={metr_s2['worst_group_error']:.4f}"
            )

        # Evaluate on test using final params
        c_test = compute_cost_for_target_rejection(post_test, class_to_group, alpha_best, mu_best, beta, float(target_rej))
        plugin.set_params(
            torch.tensor(alpha_best, dtype=torch.float32, device=DEVICE),
            torch.tensor(mu_best, dtype=torch.float32, device=DEVICE),
            torch.tensor(beta, dtype=torch.float32, device=DEVICE),
            float(c_test),
        )
        preds_test = plugin.predict(post_test)
        rej_test = plugin.reject(post_test)
        m_test = compute_metrics(preds_test, labels_test, rej_test, class_to_group, class_weights)
        print(
            f"[TEST] r={target_rej:.2f} cov={m_test['coverage']:.3f} bal={m_test['balanced_error']:.4f} wst={m_test['worst_group_error']:.4f} beta={beta}"
        )
        results.append({
            "target_rejection": float(target_rej),
            "beta": beta.tolist(),
            "alpha": alpha_best.tolist(),
            "mu": mu_best.tolist(),
            "cost_test": float(c_test),
            "test_metrics": {
                "coverage": float(m_test["coverage"]),
                "balanced_error": float(m_test["balanced_error"]),
                "worst_group_error": float(m_test["worst_group_error"]),
                "group_errors": [float(x) for x in m_test["group_errors"]],
            },
        })

    r = np.array([1.0 - r_["test_metrics"]["coverage"] for r_ in results])
    ew = np.array([r_["test_metrics"]["worst_group_error"] for r_ in results])
    eb = np.array([r_["test_metrics"]["balanced_error"] for r_ in results])
    idx = np.argsort(r)
    r, ew, eb = r[idx], ew[idx], eb[idx]
    aurc_w = float(np.trapz(ew, r)) if r.size > 1 else float(ew.mean() if ew.size else 0.0)
    aurc_b = float(np.trapz(eb, r)) if r.size > 1 else float(eb.mean() if eb.size else 0.0)

    out_json = Path(CFG.results_dir) / "ltr_plugin_ce_only_worst.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Worst-group plug-in via Algorithm 2 with inner CS plug-in (Algorithm 1)",
            "target_rejections": list(CFG.target_rejections),
            "results_per_point": results,
            "rc_curve": {
                "rejection_rates": r.tolist(),
                "worst_group_errors": ew.tolist(),
                "balanced_errors": eb.tolist(),
                "aurc_worst_group": aurc_w,
                "aurc_balanced": aurc_b,
            },
        }, f, indent=2)

    plot_path = Path(CFG.results_dir) / "ltr_rc_curves_balanced_worst_ce_only_test.png"
    plot_rc(r, ew, eb, aurc_w, aurc_b, plot_path)
    print(f"Saved results to: {out_json}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()


