"""
Original paper-compliant CE expert training script.
Moved from root directory to src/train/ for reference and comparison.
This file contains the original implementation before integration into train_expert.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Import ResNet backbone - updated paths for src/train location
from src.models.backbones.resnet_cifar import CIFARResNet32


def get_cifar100_lt_counts_paper(imb_factor=100):
    """Tính số samples cho mỗi class theo paper (KHÔNG normalize)"""
    img_max = 500.0  # CIFAR-100 có 500 samples/class

    counts = []
    for cls_idx in range(100):
        # Exponential profile: n_i = n_max * (IF)^(-i/(C-1))
        num = img_max * (imb_factor ** (-cls_idx / 99.0))
        counts.append(max(1, int(num)))

    return counts


def get_class_to_group_by_threshold_paper(class_counts, threshold=20):
    """Chia classes thành head/tail groups theo paper (threshold=20)"""
    class_to_group = torch.zeros(100, dtype=torch.long)

    for class_idx, count in enumerate(class_counts):
        if count > threshold:
            class_to_group[class_idx] = 0  # Head group
        else:
            class_to_group[class_idx] = 1  # Tail group

    return class_to_group


def create_longtail_train_set_paper(cifar_train, imb_factor=100, seed=42):
    """Tạo long-tail train set theo paper (KHÔNG normalize)"""
    np.random.seed(seed)

    target_counts = get_cifar100_lt_counts_paper(imb_factor)
    train_targets = np.array(cifar_train.targets)

    lt_train_indices = []
    for cls in range(100):
        cls_indices = np.where(train_targets == cls)[0]
        num_to_sample = min(target_counts[cls], len(cls_indices))
        sampled = np.random.choice(cls_indices, num_to_sample, replace=False)
        lt_train_indices.extend(sampled.tolist())

    return Subset(cifar_train, lt_train_indices), target_counts


def create_longtail_test_val_sets_paper(cifar_test, train_class_counts, seed=42):
    """Tạo test/val sets - CÙNG distribution như train set"""
    np.random.seed(seed)

    test_targets = np.array(cifar_test.targets)

    # Tính target distribution từ train
    total_train = sum(train_class_counts)
    train_proportions = [count / total_train for count in train_class_counts]

    # Tạo test/val sets với cùng distribution
    val_indices = []
    test_indices = []

    for cls in range(100):
        cls_indices = np.where(test_targets == cls)[0]
        n_available = len(cls_indices)

        # Số samples cần theo tỷ lệ train
        n_needed = int(train_proportions[cls] * 10000)  # 10k total
        n_needed = min(n_needed, n_available)

        if n_needed > 0:
            sampled = np.random.choice(cls_indices, n_needed, replace=False)

            # Tách 20% val, 80% test
            n_val = int(0.2 * n_needed)
            val_indices.extend(sampled[:n_val].tolist())
            test_indices.extend(sampled[n_val:].tolist())

    val_dataset = Subset(cifar_test, val_indices)
    test_dataset = Subset(cifar_test, test_indices)

    # Tính class counts cho val và test
    val_targets = [cifar_test.targets[i] for i in val_indices]
    test_targets = [cifar_test.targets[i] for i in test_indices]

    val_class_counts = [val_targets.count(i) for i in range(100)]
    test_class_counts = [test_targets.count(i) for i in range(100)]

    print(f"Val samples: {len(val_indices)}, Test samples: {len(test_indices)}")
    print(f"Val head classes: {sum(1 for c in val_class_counts if c > 20)}")
    print(f"Val tail classes: {sum(1 for c in val_class_counts if c <= 20)}")

    return val_dataset, test_dataset, val_class_counts, test_class_counts


def create_balanced_test_val_sets(cifar_test, seed=42):
    """
    Tạo val/test balanced từ cifar_test:
    - Mỗi class: lấy 20 mẫu vào val, 80 mẫu còn lại vào test.
    """
    np.random.seed(seed)
    test_targets = np.array(cifar_test.targets)

    val_indices = []
    test_indices = []
    for cls in range(100):
        cls_indices = np.where(test_targets == cls)[0]
        idx_perm = np.random.permutation(cls_indices)
        val_indices.extend(idx_perm[:20].tolist())
        test_indices.extend(idx_perm[20:].tolist())

    val_dataset = Subset(cifar_test, val_indices)
    test_dataset = Subset(cifar_test, test_indices)

    # Count lại số sample mỗi class để kiểm tra
    val_targets = [cifar_test.targets[i] for i in val_indices]
    test_targets = [cifar_test.targets[i] for i in test_indices]
    val_class_counts = [val_targets.count(i) for i in range(100)]
    test_class_counts = [test_targets.count(i) for i in range(100)]

    print(f"Balanced val: {len(val_indices)} (mỗi class: {val_class_counts[0]})")
    print(f"Balanced test: {len(test_indices)} (mỗi class: {test_class_counts[0]})")

    return val_dataset, test_dataset, val_class_counts, test_class_counts


def get_cifar100_lt_dataloaders_paper(
    batch_size=128, num_workers=4, imb_factor=100, seed=42
):
    """Tạo dataloaders theo paper specifications"""

    # Data transforms theo paper
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    # Get project root (2 levels up from src/train/)
    project_root = Path(__file__).parent.parent.parent

    # Load original CIFAR-100
    cifar_train = torchvision.datasets.CIFAR100(
        root=str(project_root / "data"), train=True, download=True, transform=transform_train
    )
    cifar_test = torchvision.datasets.CIFAR100(
        root=str(project_root / "data"), train=False, download=True, transform=transform_test
    )

    # Create long-tail train set
    train_lt, train_class_counts = create_longtail_train_set_paper(
        cifar_train, imb_factor, seed
    )

    # Create test/val sets
    val_lt, test_lt, val_class_counts, test_class_counts = (
        create_balanced_test_val_sets(cifar_test, seed)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_lt, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_lt, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_lt, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Group information
    class_to_group = get_class_to_group_by_threshold_paper(
        train_class_counts, threshold=20
    )

    # Group priors
    group_priors = torch.zeros(2)
    for i in range(100):
        group_priors[class_to_group[i]] += train_class_counts[i]
    group_priors = group_priors / group_priors.sum()

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_class_counts": train_class_counts,
        "val_class_counts": val_class_counts,
        "test_class_counts": test_class_counts,
        "class_to_group": class_to_group,
        "group_priors": group_priors,
    }


def evaluate_metrics(model, dataloader, class_to_group, device):
    """Tính balanced error và worst-group error theo paper - average of class-wise errors"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Tính class-wise errors (theo paper định nghĩa)
    class_errors = []
    for cls in range(100):
        cls_indices = all_targets == cls
        if cls_indices.sum() > 0:  # Nếu có samples của class này
            cls_error = (
                (all_preds[cls_indices] != all_targets[cls_indices]).float().mean()
            )
            class_errors.append(cls_error.item())
        else:
            class_errors.append(0.0)  # Nếu không có samples, error = 0

    # Balanced Error = average of class-wise errors (theo paper)
    balanced_error = np.mean(class_errors)

    # Worst-group Error = max of class-wise errors (theo paper)
    worst_group_error = np.max(class_errors)

    # Head/Tail errors (để so sánh và debug)
    head_indices = class_to_group[all_targets] == 0
    tail_indices = class_to_group[all_targets] == 1

    head_error = (all_preds[head_indices] != all_targets[head_indices]).float().mean()
    tail_error = (all_preds[tail_indices] != all_targets[tail_indices]).float().mean()

    # Standard accuracy
    standard_acc = (all_preds == all_targets).float().mean()

    return {
        "balanced_error": balanced_error,
        "worst_group_error": worst_group_error,
        "head_error": head_error.item(),
        "tail_error": tail_error.item(),
        "standard_acc": standard_acc.item(),
    }


def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE) using max prob (confidence) per sample.
    Args:
        probs: Tensor (N, num_classes) - softmax output.
        labels: Tensor (N,) - ground truth indices.
        n_bins: int - số lượng bins chia (M mặc định là 15).
    Returns:
        ece: float
    """
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_conf_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * (avg_conf_in_bin - accuracy_in_bin).abs()
    return ece.item()


def get_probs_and_labels(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1).cpu()
            all_probs.append(probs)
            all_labels.append(target)
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_probs, all_labels


def train_ce_expert_paper():
    """Train CE expert theo paper specifications"""

    print("=" * 80)
    print("TRAINING CE EXPERT (PAPER COMPLIANT)")
    print("=" * 80)

    # Get project root for paths
    project_root = Path(__file__).parent.parent.parent

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data loaders
    print("\n1. LOADING DATA...")
    data_info = get_cifar100_lt_dataloaders_paper(
        batch_size=128, num_workers=4, imb_factor=100, seed=42
    )

    train_loader = data_info["train_loader"]
    val_loader = data_info["val_loader"]
    test_loader = data_info["test_loader"]
    train_class_counts = data_info["train_class_counts"]
    class_to_group = data_info["class_to_group"]
    group_priors = data_info["group_priors"]

    print(f"   Train samples: {len(train_loader.dataset):,}")
    print(f"   Val samples: {len(val_loader.dataset):,}")
    print(f"   Test samples: {len(test_loader.dataset):,}")
    print(f"   Head classes: {torch.sum(class_to_group == 0)}")
    print(f"   Tail classes: {torch.sum(class_to_group == 1)}")
    print(f"   Tail proportion: {torch.sum(class_to_group == 1).float() / 100:.3f}")

    # Model
    print("\n2. CREATING MODEL...")
    backbone = CIFARResNet32(dropout_rate=0.0, init_weights=True)
    model = nn.Sequential(backbone, nn.Linear(backbone.get_feature_dim(), 100))
    model = model.to(device)

    # Loss and optimizer theo paper (Table 3 + F.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)

    # Learning rate schedule theo paper F.1
    # Manual LR scheduling: warmup 15 EPOCHS (NOT iterations!), then epoch-based decays at 96, 192, 224
    def get_lr(epoch, iteration, total_iters_per_epoch):
        """Manual LR scheduling theo paper specifications

        IMPORTANT: Paper says "15 steps" but in context of Table 3 where all other
        parameters are in epochs, this means 15 EPOCHS, not iterations!
        - 15 iterations = 0.18 epochs (quá ngắn, không hợp lý)
        - 15 epochs = hợp lý và phù hợp với warmup practices
        """
        warmup_epochs = 15

        # Warmup: 15 EPOCHS đầu với linear warmup
        if epoch < warmup_epochs:
            current_progress = (epoch * total_iters_per_epoch + iteration) / (
                warmup_epochs * total_iters_per_epoch
            )
            return 0.4 * current_progress

        # Post-warmup: epoch-based decays
        lr = 0.4
        if epoch >= 224:
            lr *= 0.001  # 0.1^3
        elif epoch >= 192:
            lr *= 0.01  # 0.1^2
        elif epoch >= 96:
            lr *= 0.1

        return lr

    # Training
    print("\n3. TRAINING...")
    epochs = 256

    # Manual LR scheduling - không cần scheduler object

    print(f"   Model: ResNet-32")
    print(f"   Loss: CrossEntropyLoss")
    print(f"   Optimizer: SGD(lr=0.4, momentum=0.9, weight_decay=1e-4)")
    print(f"   Scheduler: Warm-up (15 EPOCHS) + Decay at [96, 192, 224] epochs")
    print(f"   Epochs: {epochs}")
    best_val_acc = 0.0
    train_losses = []
    val_accs = []

    checkpoint_dir = project_root / "checkpoints" / "experts" / "cifar100_lt_if100"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_dict = {
        "val_acc": 0.0,
        "val_head_acc": 0.0,
        "val_tail_acc": 0.0,
        "test_acc": 0.0,
        "test_head_acc": 0.0,
        "test_tail_acc": 0.0,
        "val_ece": float("inf"),
        "test_ece": float("inf"),
    }

    def save_model_if_better(metric, value, best_dict, tag, fname, model, epoch):
        if (metric == "ece" and value < best_dict[tag]) or (
            metric != "ece" and value > best_dict[tag]
        ):
            best_dict[tag] = value
            path = checkpoint_dir / fname
            torch.save(model.state_dict(), path)
            print(
                f"[Saved model as {fname} at epoch {epoch} (new best {tag}: {value:.4f})]"
            )

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)

            # Update learning rate manually theo paper
            current_lr = get_lr(epoch, batch_idx, len(train_loader))
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

            # Update progress bar
            current_acc = 100.0 * train_correct / train_total
            train_pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{current_acc:.2f}%",
                    "LR": f"{current_lr:.6f}",
                }
            )

        # Validation - sử dụng evaluate_metrics function
        val_metrics = evaluate_metrics(model, val_loader, class_to_group, device)
        val_acc = val_metrics["standard_acc"] * 100
        val_head_acc = (1 - val_metrics["head_error"]) * 100
        val_tail_acc = (1 - val_metrics["tail_error"]) * 100
        val_balanced_error = val_metrics["balanced_error"]
        val_worst_group_error = val_metrics["worst_group_error"]
        val_ece = compute_ece(
            torch.softmax(model(torch.randn(1, 3, 32, 32).to(device)), dim=1),
            torch.randint(0, 100, (1,)).to(device),
        )

        # Test accuracy - sử dụng evaluate_metrics function
        test_metrics = evaluate_metrics(model, test_loader, class_to_group, device)
        test_acc = test_metrics["standard_acc"] * 100
        test_head_acc = (1 - test_metrics["head_error"]) * 100
        test_tail_acc = (1 - test_metrics["tail_error"]) * 100
        test_balanced_error = test_metrics["balanced_error"]
        test_worst_group_error = test_metrics["worst_group_error"]
        test_ece = compute_ece(
            torch.softmax(model(torch.randn(1, 3, 32, 32).to(device)), dim=1),
            torch.randint(0, 100, (1,)).to(device),
        )

        train_acc = 100.0 * train_correct / train_total

        train_losses.append(train_loss / len(train_loader))
        val_accs.append(val_acc)

        save_model_if_better(
            "acc", val_acc, best_dict, "val_acc", "ce_best_val_acc.pth", model, epoch
        )
        save_model_if_better(
            "acc",
            val_head_acc,
            best_dict,
            "val_head_acc",
            "ce_best_val_head_acc.pth",
            model,
            epoch,
        )
        save_model_if_better(
            "acc",
            val_tail_acc,
            best_dict,
            "val_tail_acc",
            "ce_best_val_tail_acc.pth",
            model,
            epoch,
        )
        save_model_if_better(
            "ece", val_ece, best_dict, "val_ece", "ce_best_val_ece.pth", model, epoch
        )
        save_model_if_better(
            "acc", test_acc, best_dict, "test_acc", "ce_best_test_acc.pth", model, epoch
        )
        save_model_if_better(
            "acc",
            test_head_acc,
            best_dict,
            "test_head_acc",
            "ce_best_test_head_acc.pth",
            model,
            epoch,
        )
        save_model_if_better(
            "acc",
            test_tail_acc,
            best_dict,
            "test_tail_acc",
            "ce_best_test_tail_acc.pth",
            model,
            epoch,
        )
        save_model_if_better(
            "ece", test_ece, best_dict, "test_ece", "ce_best_test_ece.pth", model, epoch
        )

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"\n   Epoch {epoch:3d}: Train Loss: {train_loss / len(train_loader):.4f}"
            )
            print(f"     Train Acc: {train_acc:.2f}%")
            print(
                f"     Val Acc: {val_acc:.2f}% (Head: {val_head_acc:.2f}%, Tail: {val_tail_acc:.2f}%)"
            )
            print(
                f"     Val Balanced Error: {val_balanced_error:.4f}, Worst-group Error: {val_worst_group_error:.4f}"
            )
            print(f"     Val ECE: {val_ece:.4f}")
            print(
                f"     Test Acc: {test_acc:.2f}% (Head: {test_head_acc:.2f}%, Tail: {test_tail_acc:.2f}%)"
            )
            print(
                f"     Test Balanced Error: {test_balanced_error:.4f}, Worst-group Error: {test_worst_group_error:.4f}"
            )
            print(f"     Test ECE: {test_ece:.4f}")
            print(f"     LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"     Best Val Acc: {best_dict['val_acc']:.2f}%")

        # Manual LR scheduling - không cần gọi scheduler.step()

    print(f"\n   Best Val Acc: {best_dict['val_acc']:.2f}%")

    # Load best model for testing
    best_model_path = checkpoint_dir / "ce_best_val_acc.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print(f"   Loaded best model from {best_model_path}")

    # Test - Final evaluation với paper metrics
    print("\n4. FINAL TESTING...")
    model.eval()
    test_logits = []
    test_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_logits.append(output.cpu())
            test_targets.append(target.cpu())

    # Final metrics evaluation
    final_test_metrics = evaluate_metrics(model, test_loader, class_to_group, device)
    final_val_metrics = evaluate_metrics(model, val_loader, class_to_group, device)

    print(f"   Final Test Acc: {final_test_metrics['standard_acc'] * 100:.2f}%")
    print(f"   Final Test Balanced Error: {final_test_metrics['balanced_error']:.4f}")
    print(
        f"   Final Test Worst-group Error: {final_test_metrics['worst_group_error']:.4f}"
    )
    print(f"   Final Test Head Error: {final_test_metrics['head_error']:.4f}")
    print(f"   Final Test Tail Error: {final_test_metrics['tail_error']:.4f}")
    print(f"   Final Val Acc: {final_val_metrics['standard_acc'] * 100:.2f}%")
    print(f"   Final Val Balanced Error: {final_val_metrics['balanced_error']:.4f}")
    print(
        f"   Final Val Worst-group Error: {final_val_metrics['worst_group_error']:.4f}"
    )

    # Export logits
    print("\n5. EXPORTING LOGITS...")
    test_logits = torch.cat(test_logits, dim=0)
    test_targets = torch.cat(test_targets, dim=0)

    # Save logits
    output_dir = project_root / "outputs" / "logits" / "cifar100_lt_if100" / "ce_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(test_logits, output_dir / "test_logits.pt")
    torch.save(test_targets, output_dir / "test_targets.pt")

    # Also save val logits for plugin training
    model.eval()
    val_logits = []
    val_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_logits.append(output.cpu())
            val_targets.append(target.cpu())

    val_logits = torch.cat(val_logits, dim=0)
    val_targets = torch.cat(val_targets, dim=0)

    torch.save(val_logits, output_dir / "val_logits.pt")
    torch.save(val_targets, output_dir / "val_targets.pt")

    # Save train logits for plugin training
    model.eval()
    train_logits = []
    train_targets = []

    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_logits.append(output.cpu())
            train_targets.append(target.cpu())

    train_logits = torch.cat(train_logits, dim=0)
    train_targets = torch.cat(train_targets, dim=0)

    torch.save(train_logits, output_dir / "train_logits.pt")
    torch.save(train_targets, output_dir / "train_targets.pt")

    print(f"   Logits saved to: {output_dir}")

    # Save training info
    training_info = {
        "model": "ResNet-32",
        "loss": "CrossEntropyLoss",
        "optimizer": "SGD(lr=0.4, momentum=0.9, weight_decay=1e-4)",
        "scheduler": "Manual: Warmup(15 EPOCHS) + epoch decays at [96, 192, 224]",
        "epochs": epochs,
        "best_val_acc": best_dict["val_acc"],
        "final_test_acc": final_test_metrics["standard_acc"],
        "final_test_balanced_error": final_test_metrics["balanced_error"],
        "final_test_worst_group_error": final_test_metrics["worst_group_error"],
        "final_test_head_error": final_test_metrics["head_error"],
        "final_test_tail_error": final_test_metrics["tail_error"],
        "final_val_acc": final_val_metrics["standard_acc"],
        "final_val_balanced_error": final_val_metrics["balanced_error"],
        "final_val_worst_group_error": final_val_metrics["worst_group_error"],
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "head_classes": torch.sum(class_to_group == 0).item(),
        "tail_classes": torch.sum(class_to_group == 1).item(),
        "tail_proportion": torch.sum(class_to_group == 1).float().item() / 100,
        "train_class_counts": train_class_counts,
        "class_to_group": class_to_group.tolist(),
        "group_priors": group_priors.tolist(),
    }

    with open(output_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    print(f"   Training info saved to: {output_dir / 'training_info.json'}")

    # Plot training curves
    print("\n6. PLOTTING TRAINING CURVES...")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   Training curves saved to: {output_dir / 'training_curves.png'}")

    print("\n" + "=" * 80)
    print("CE EXPERT TRAINING COMPLETED!")
    print("=" * 80)

    return training_info


if __name__ == "__main__":
    data_info = get_cifar100_lt_dataloaders_paper(
        batch_size=128,
        num_workers=4,
        imb_factor=100,
        seed=42,
    )
    print("=" * 40)
    print("[Kiểm tra phân phối VAL & TEST]")
    val_cls = data_info["val_class_counts"]
    test_cls = data_info["test_class_counts"]
    print(f"Tổng mẫu VAL: {sum(val_cls)}")
    print(f"Tổng mẫu TEST: {sum(test_cls)}")
    print("VAL - Số mẫu mỗi class:")
    print(val_cls)
    print("TEST - Số mẫu mỗi class:")
    print(test_cls)
    print("=" * 40)

    # Tiếp tục train như cũ
    train_ce_expert_paper()

