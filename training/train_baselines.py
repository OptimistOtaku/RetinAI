"""
Baseline model comparison training script.

Trains standard SOTA architectures (ResNet-50, VGG-16, DenseNet-121,
MobileNetV2, EfficientNet-B0) using 3-channel RGB inputs — **no diffusion
anomaly guidance** — on the same task, dataset, and splits as our primary
EfficientNet-B3 4-channel models.

Running this script lets you measure the exact accuracy/AUC gain from the
diffusion-based anomaly channel in a controlled, apples-to-apples setting.

Usage
-----
# Diabetic Retinopathy (5-class)
python training/train_baselines.py \\
    --task cnn1_dr \\
    --labels_file <eyepacs_anomaly.csv> \\
    --output_dir models/checkpoints/baselines

# Glaucoma (binary)
python training/train_baselines.py \\
    --task cnn2_glaucoma \\
    --labels_file <glaucoma_anomaly.csv> \\
    --output_dir models/checkpoints/baselines

# Pathologic Myopia (binary)
python training/train_baselines.py \\
    --task cnn3_pm \\
    --labels_file <palm_anomaly.csv> \\
    --output_dir models/checkpoints/baselines

Results are saved to <output_dir>/baseline_comparison_<task>.json.
"""

import argparse
import json
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.datasets import EyePACSDataset, REFUGEDataset, PALMDataset
from src.utils.checkpoint import save_checkpoint
from src.utils.reproducibility import set_random_seeds, setup_logging
from src.utils.transforms import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Baseline architectures to evaluate — all trained in 3ch RGB mode.
# Using timm model names so the same hub weights are resolved consistently.
# ---------------------------------------------------------------------------
BASELINE_ARCHITECTURES = [
    "resnet50",
    "vgg16",
    "densenet121",
    "mobilenetv2_100",
    "efficientnet_b0",
]

# Task metadata
TASK_CONFIG = {
    "cnn1_dr": {
        "num_classes": 5,
        "class_names": ["Grade0", "Grade1", "Grade2", "Grade3", "Grade4"],
        "dataset_cls": EyePACSDataset,
        "split_mode": "random",   # random 80/20 split (same as train_cnn1_dr.py)
        "binary": False,
    },
    "cnn2_glaucoma": {
        "num_classes": 2,
        "class_names": ["non_glaucoma", "glaucoma"],
        "dataset_cls": REFUGEDataset,
        "split_mode": "csv",      # uses 'split' column in CSV (train / test)
        "binary": True,
    },
    "cnn3_pm": {
        "num_classes": 2,
        "class_names": ["non_pm", "pm"],
        "dataset_cls": PALMDataset,
        "split_mode": "csv",      # uses 'split' column in CSV (train / val)
        "binary": True,
    },
}


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_baseline_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create a baseline architecture using timm with standard 3-channel input.

    Args:
        arch:        timm model name (e.g. 'resnet50', 'vgg16').
        num_classes: Number of output classes.
        pretrained:  Whether to use pretrained ImageNet weights.

    Returns:
        PyTorch model ready for training on 3-channel RGB images.
    """
    model = timm.create_model(
        arch,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    for param in model.parameters():
        param.requires_grad = True
    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        correct += (preds == labels.cpu().numpy()).sum()
        total += labels.size(0)

    probs_arr = np.concatenate(all_probs)
    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)

    return total_loss / len(loader), correct / total, probs_arr, preds_arr, labels_arr


def compute_metrics(labels, preds, probs, num_classes, binary):
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    try:
        if binary:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            labels_bin = label_binarize(labels, classes=list(range(num_classes)))
            auc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = None

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "auc_roc": float(auc) if auc is not None else None,
    }


# ---------------------------------------------------------------------------
# Dataset / split helpers
# ---------------------------------------------------------------------------

def build_dataloaders(task: str, labels_file: str, batch_size: int,
                      num_workers: int, seed: int):
    cfg = TASK_CONFIG[task]
    DatasetCls = cfg["dataset_cls"]

    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    if cfg["split_mode"] == "random":
        full_ds = DatasetCls(labels_file, transform=val_transform, use_4ch=False, split=None)
        n_total = len(full_ds)
        n_val = int(0.2 * n_total)
        n_train = n_total - n_val

        train_idx, val_idx = torch.utils.data.random_split(
            range(n_total),
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )

        train_ds_base = DatasetCls(labels_file, transform=train_transform, use_4ch=False, split=None)
        val_ds_base = DatasetCls(labels_file, transform=val_transform, use_4ch=False, split=None)

        train_ds = Subset(train_ds_base, train_idx.indices)
        val_ds = Subset(val_ds_base, val_idx.indices)

    else:
        # Use CSV split column
        train_split = "train"
        val_split = "test" if task == "cnn2_glaucoma" else "val"

        train_ds = DatasetCls(labels_file, transform=train_transform, use_4ch=False, split=train_split)
        val_ds = DatasetCls(labels_file, transform=val_transform, use_4ch=False, split=val_split)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Per-architecture training
# ---------------------------------------------------------------------------

def train_architecture(arch, task, args, logger):
    cfg = TASK_CONFIG[task]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Architecture: {arch}  |  Task: {task}")
    logger.info(f"{'=' * 60}")

    # Data
    train_loader, val_loader = build_dataloaders(
        task, args.labels_file, args.batch_size, args.num_workers, args.seed
    )
    logger.info(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    # Model
    model = create_baseline_model(arch, cfg["num_classes"], pretrained=not args.no_pretrain)
    model.to(args.device)

    # Loss — weighted cross-entropy for class imbalance
    criterion = nn.CrossEntropyLoss()

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(args.device.split(":")[0])

    best_val_acc = 0.0
    best_metrics = {}
    epochs_no_improve = 0
    ckpt_dir = os.path.join(args.output_dir, task, arch)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{arch}_best.pth")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, scaler
        )
        val_loss, val_acc, probs, preds, labels = evaluate(
            model, val_loader, criterion, args.device
        )
        scheduler.step()

        logger.info(
            f"Epoch {epoch + 1:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_metrics = compute_metrics(
                labels, preds, probs, cfg["num_classes"], cfg["binary"]
            )
            save_checkpoint(
                model, optimizer, epoch,
                {"best_val_acc": best_val_acc},
                ckpt_path, scheduler,
            )
            logger.info(f"  ✓ New best val_acc={best_val_acc:.4f} — checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"  Early stopping after {epoch + 1} epochs")
                break

    logger.info(f"\nBest val acc: {best_val_acc:.4f}")
    logger.info(f"Metrics: {best_metrics}")
    best_metrics["checkpoint"] = ckpt_path
    return best_metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SOTA baseline architectures for within-study comparison"
    )
    parser.add_argument(
        "--task", required=True, choices=list(TASK_CONFIG),
        help="Which disease task to train baselines for",
    )
    parser.add_argument("--labels_file", required=True, help="Path to anomaly CSV")
    parser.add_argument("--output_dir", default="models/checkpoints/baselines")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max epochs per architecture (early stopping may stop sooner)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience in epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--no_pretrain", action="store_true",
        help="Disable ImageNet pretrained weights (not recommended)",
    )
    parser.add_argument(
        "--architectures", nargs="+", default=BASELINE_ARCHITECTURES,
        help="Subset of architectures to train (default: all five)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_random_seeds(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(
        log_dir=args.output_dir,
        log_filename=f"baseline_training_{args.task}.log",
    )

    logger.info("RetinAI — Baseline Architecture Comparison")
    logger.info(f"Task:    {args.task}")
    logger.info(f"Device:  {args.device}")
    logger.info(f"Epochs:  {args.epochs} (patience={args.patience})")
    logger.info(f"Architectures: {args.architectures}")

    comparison = {}
    for arch in args.architectures:
        try:
            metrics = train_architecture(arch, args.task, args, logger)
            comparison[arch] = {"status": "ok", **metrics}
        except Exception as exc:
            logger.error(f"Training failed for {arch}: {exc}")
            comparison[arch] = {"status": "error", "error": str(exc)}

    # Save comparison JSON
    results_path = os.path.join(args.output_dir, f"baseline_comparison_{args.task}.json")
    with open(results_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"\nComparison results saved → {results_path}")

    # Pretty-print summary table
    logger.info("\n" + "=" * 70)
    logger.info(f"{'Architecture':<20} {'Accuracy':>10} {'F1 Macro':>10} {'AUC-ROC':>10}")
    logger.info("-" * 70)
    for arch, m in comparison.items():
        if m.get("status") == "ok":
            logger.info(
                f"{arch:<20} {m.get('accuracy', 0):>10.4f} "
                f"{m.get('f1_macro', 0):>10.4f} "
                f"{m.get('auc_roc') or 0:>10.4f}"
            )
        else:
            logger.info(f"{arch:<20} {'ERROR':>10}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
