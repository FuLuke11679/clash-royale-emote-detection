# src/train.py

import os
import argparse
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score

from src.data.fer_dataset import get_dataloaders
from src.data.fer_dataset import NUM_CLASSES
from src.models.resnet_emotion import EmotionResNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train FER2013 Emotion ResNet (folder-based)")
    parser.add_argument("--train_dir", type=str, default="data/face/train")
    parser.add_argument("--test_dir", type=str, default="data/face/test")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="models/checkpoints",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    return parser.parse_args()


def build_optimizer(
    model: torch.nn.Module,
    opt_name: str,
    lr: float,
    weight_decay: float,
):
    if opt_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    model.train()
    epoch_losses = []
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = float(np.mean(epoch_losses))
    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average="macro"))

    return {"loss": avg_loss, "acc": acc, "f1": f1}


def eval_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    epoch_losses = []
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            epoch_losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = float(np.mean(epoch_losses))
    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average="macro"))

    return {"loss": avg_loss, "acc": acc, "f1": f1}


def main():
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
    )

    # Infer number of classes from the dataset to avoid mismatch with CLASS_NAMES constant
    num_classes = len(train_loader.dataset.classes)
    if num_classes != NUM_CLASSES:
        print(f"[WARN] Dataset contains {num_classes} classes but NUM_CLASSES={NUM_CLASSES}. Using inferred value {num_classes}.")

    model = EmotionResNet(
        backbone=args.backbone,
        num_classes=num_classes,
        pretrained=True,
        dropout_p=args.dropout_p,
    ).to(device)

    # Class weights for imbalance using train_loader subset
    class_counts = np.bincount(train_loader.dataset.targets, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_resnet.pt")

    # History for training curves and CSV logging
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(
            f"  Train   | loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['acc']:.4f}, f1={train_metrics['f1']:.4f}"
        )

        val_metrics = eval_one_epoch(model, val_loader, criterion, device)
        print(
            f"  Val     | loss={val_metrics['loss']:.4f}, "
            f"acc={val_metrics['acc']:.4f}, f1={val_metrics['f1']:.4f}"
        )

        # record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1"].append(val_metrics["f1"])

        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  --> New best model saved to {best_ckpt_path}")
        else:
            patience_counter += 1
            print(f"  No improvement, patience={patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining finished. Best epoch: {best_epoch} with val loss={best_val_loss:.4f}")

    # Optional: Evaluate best model on test set
    print("Loading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.to(device)

    test_metrics = eval_one_epoch(model, test_loader, criterion, device)
    print(
        f"Test    | loss={test_metrics['loss']:.4f}, "
        f"acc={test_metrics['acc']:.4f}, f1={test_metrics['f1']:.4f}"
    )

    # Save training metrics to CSV
    metrics_csv = os.path.join(args.checkpoint_dir, "metrics.csv")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"])
        for i in range(len(history["epoch"])):
            writer.writerow(
                [
                    history["epoch"][i],
                    history["train_loss"][i],
                    history["train_acc"][i],
                    history["train_f1"][i],
                    history["val_loss"][i],
                    history["val_acc"][i],
                    history["val_f1"][i],
                ]
            )
    print(f"Saved training metrics to {metrics_csv}")

    # Plot training curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
        axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
        axes[0].set_title("Loss")
        axes[0].legend()

        axes[1].plot(history["epoch"], history["train_acc"], label="train_acc")
        axes[1].plot(history["epoch"], history["val_acc"], label="val_acc")
        axes[1].set_title("Accuracy")
        axes[1].legend()

        plot_path = os.path.join(args.checkpoint_dir, "training_curves.png")
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved training curves to {plot_path}")
    except Exception as e:
        print(f"Warning: failed to save training plots: {e}")


if __name__ == "__main__":
    main()
