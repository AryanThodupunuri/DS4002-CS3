#!/usr/bin/env python3
"""
Baseline CNN training script for Extreme vs Normal weather classification.

Features:
- Loads images from a split directory (train/val/test) if provided, else splits from a single folder of class subdirs.
- Minimal custom CNN (3 conv blocks) suitable for 224x224 RGB images.
- Training loop with validation, early best checkpoint saving.
- Optional dry-run mode to sanity-check data/model without full training.
- Optional eval-only mode to regenerate confusion matrix & classification report from an existing checkpoint without retraining.
- Saves: best model weights (.pth), metrics CSV, and confusion matrix PNG.

Usage examples:
  python SCRIPTS/train_baseline_cnn.py \
      --data-root DATA/raw_data/cleaned_data \
      --epochs 10 --batch-size 64 --out-dir MODELS

  # If you already have a split folder with train/val/test
  python SCRIPTS/train_baseline_cnn.py \
      --split-root dataset_split \
      --epochs 10 --batch-size 64 --out-dir MODELS

  # Dry-run to verify everything works (one forward/backward pass)
  python SCRIPTS/train_baseline_cnn.py --dry-run --limit-per-class 16

    # Eval-only: load existing checkpoint and (re)generate confusion matrix/report
    python SCRIPTS/train_baseline_cnn.py --eval-only --model-path MODELS/baseline_cnn_best.pth --split-root dataset_split
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

import torchvision
from torchvision import transforms

try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception:
    classification_report = None  # type: ignore
    confusion_matrix = None  # type: ignore

import matplotlib.pyplot as plt
import csv


CLASSES = ("extreme", "normal")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Input: 3x224x224
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # Scale to [0,1]; no mean/std normalization for baseline simplicity
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return train_tf, val_tf


def make_datasets(
    data_root: Path,
    split_root: Optional[Path],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    limit_per_class: Optional[int] = None,
    img_size: int = 224,
) -> Tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder]:
    """
    If split_root exists with train/val/test, use those.
    Else, load from data_root (class-subfolder structure) and create random splits.
    """
    train_tf, val_tf = build_transforms(img_size)

    if split_root and (split_root / "train").exists() and (split_root / "val").exists():
        train_ds = torchvision.datasets.ImageFolder(split_root / "train", transform=train_tf)
        val_ds = torchvision.datasets.ImageFolder(split_root / "val", transform=val_tf)
        test_ds = torchvision.datasets.ImageFolder(split_root / "test", transform=val_tf) if (split_root / "test").exists() else val_ds
        return train_ds, val_ds, test_ds

    # Fallback: single folder -> split
    full_ds = torchvision.datasets.ImageFolder(data_root, transform=train_tf)

    # Optional class-wise limiting
    if limit_per_class is not None and limit_per_class > 0:
        indices_by_class: Dict[int, List[int]] = {i: [] for i in range(len(full_ds.classes))}
        for idx, (_, label) in enumerate(full_ds.samples):
            if len(indices_by_class[label]) < limit_per_class:
                indices_by_class[label].append(idx)
        limited_indices: List[int] = []
        for lbl in sorted(indices_by_class.keys()):
            limited_indices.extend(indices_by_class[lbl])
        full_ds = Subset(full_ds, limited_indices)  # type: ignore

    n = len(full_ds)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    # Ensure val/test use eval transforms (resize + ToTensor only)
    def set_subset_transform(sub: Subset, tf):  # type: ignore
        if isinstance(sub.dataset, torchvision.datasets.ImageFolder):
            sub.dataset.transform = tf
        else:
            # Subset of ImageFolder
            sub.dataset.dataset.transform = tf  # type: ignore[attr-defined]

    set_subset_transform(train_ds, train_tf)
    set_subset_transform(val_ds, val_tf)
    set_subset_transform(test_ds, val_tf)

    return train_ds, val_ds, test_ds


def train_one_epoch(model, loader, device, optimizer, criterion) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return running_loss / max(total, 1), correct / max(total, 1), all_preds, all_labels


def save_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str], out_path: Path):
    if confusion_matrix is None:
        return
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train a baseline CNN for extreme vs normal classification")
    parser.add_argument("--data-root", type=str, default=str(Path("DATA")/"raw_data"/"cleaned_data"), help="Root folder with class subdirectories (used if split-root not provided)")
    parser.add_argument("--split-root", type=str, default="", help="Optional split root with train/val/test subfolders")
    parser.add_argument("--out-dir", type=str, default="MODELS", help="Directory to save models and metrics")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-per-class", type=int, default=0, help="If >0, limits images per class for quick runs")
    parser.add_argument("--dry-run", action="store_true", help="Run a single training step to sanity-check setup")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; load checkpoint and produce confusion matrix/report")
    parser.add_argument("--model-path", type=str, default=str(Path("MODELS")/"baseline_cnn_best.pth"), help="Checkpoint to load in eval-only mode (or after training)")
    parser.add_argument("--eval-suffix", type=str, default="eval", help="Suffix for confusion matrix/report filenames in eval-only mode")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_root = Path(args.split_root) if args.split_root else None
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    report_dir = Path("reports")/"training"
    ensure_dir(report_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    limit = args.limit_per_class if args.limit_per_class and args.limit_per_class > 0 else None
    train_ds, val_ds, test_ds = make_datasets(
        data_root=data_root,
        split_root=split_root,
        val_ratio=0.15,
        test_ratio=0.15,
        limit_per_class=limit,
        img_size=args.img_size,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = BaselineCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -------- EVAL-ONLY SHORT CIRCUIT --------
    if args.eval_only:
        ckpt_path = Path(args.model_path)
        if not ckpt_path.exists():
            raise SystemExit(f"--eval-only specified but checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        elif isinstance(ckpt, dict):
            # attempt direct state_dict
            try:
                model.load_state_dict(ckpt)
            except Exception as e:
                raise SystemExit(f"Failed to load checkpoint state dict: {e}")
        else:
            raise SystemExit("Unsupported checkpoint format for eval-only mode")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, device, criterion)
        print(f"Eval-only | Test: loss={test_loss:.4f} acc={test_acc:.4f}")
        cm_name = f"baseline_confusion_matrix_{args.eval_suffix}.png"
        rpt_name = f"baseline_classification_report_{args.eval_suffix}.txt"
        save_confusion_matrix(y_true, y_pred, class_names=list(CLASSES), out_path=report_dir/cm_name)
        if classification_report is not None:
            rep = classification_report(y_true, y_pred, target_names=list(CLASSES))
            (report_dir/rpt_name).write_text(rep)
            print(f"Wrote classification report: {report_dir/rpt_name}")
        else:
            print("sklearn unavailable; skipping classification report.")
        print("Eval-only complete.")
        return

    if args.dry_run:
        # One step train/eval to validate everything is wired
        model.train()
        xb, yb = next(iter(train_loader))
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print("Dry-run successful. Loss=", float(loss.item()))
        return

    best_val_acc = 0.0
    best_model_path = Path(args.model_path) if args.model_path else out_dir / "baseline_cnn_best.pth"
    metrics_csv = out_dir / "baseline_metrics.csv"

    with metrics_csv.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])
            print(f"Epoch {epoch:02d}/{args.epochs} | train {train_loss:.4f}/{train_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state': model.state_dict(),
                    'epoch': epoch,
                    'val_acc': best_val_acc,
                    'config': vars(args),
                }, best_model_path)

    # Load best and evaluate on test set
    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])

    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, device, criterion)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    # Save confusion matrix
    save_confusion_matrix(y_true, y_pred, class_names=list(CLASSES), out_path=report_dir/"baseline_confusion_matrix.png")

    # Save text report if sklearn is available
    report_txt = report_dir / "baseline_classification_report.txt"
    if classification_report is not None:
        rep = classification_report(y_true, y_pred, target_names=list(CLASSES))
        report_txt.write_text(rep)

    print(f"Training complete. Best model: {best_model_path} | Metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
