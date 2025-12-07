import os
import argparse
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, f1_score

from src.data.fer_dataset import get_dataloaders
from src.models.baseline_cnn import BaselineCNN
import csv
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline CNN on FER-2013 (folder-based)")
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--test_dir", type=str, default="data/test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="sgd_nesterov", choices=["adamw", "sgd", "sgd_nesterov"])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="models/checkpoints",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_acc"],
        help="Which metric to monitor for scheduler/early-stopping (val_loss or val_acc)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable automatic mixed precision when CUDA is available",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--balanced_sampler",
        action="store_true",
        help="Use a WeightedRandomSampler to balance classes during training (oversample minorities)",
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable class-weighted loss (useful when also using a balanced sampler)",
    )
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, opt_name: str, lr: float, weight_decay: float):
    if opt_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == "sgd_nesterov":
        return SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, Any]:
    model.train()
    epoch_losses = []
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training when scaler is provided and CUDA is available
        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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


def eval_one_epoch(model: torch.nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, Any]:
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

    # Resolve dataset paths: accept either data/train + data/test or data/face/train + data/face/test
    train_dir = args.train_dir
    test_dir = args.test_dir
    if not os.path.isdir(train_dir):
        alt = os.path.join("data", "face", "train")
        if os.path.isdir(alt):
            print(f"Note: using dataset under {alt}")
            train_dir = alt
    if not os.path.isdir(test_dir):
        alt = os.path.join("data", "face", "test")
        if os.path.isdir(alt):
            print(f"Note: using dataset under {alt}")
            test_dir = alt

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        msg = (
            "Could not find dataset directories. Tried:\n"
            f" - {args.train_dir}\n"
            f" - data/face/train\n"
            f" - {args.test_dir}\n"
            f" - data/face/test\n"
            "Please set --train_dir and --test_dir to valid ImageFolder-style folders.")
        raise FileNotFoundError(msg)

    # Respect TRAIN_AUGMENT env var if set (hp_search uses this to disable augmentation)
    env_train_augment = os.environ.get("TRAIN_AUGMENT")
    if env_train_augment is not None:
        train_augment = False if env_train_augment == "0" else True
        print(f"TRAIN_AUGMENT env var set -> train_augment={train_augment}")
    else:
        train_augment = True

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        train_augment=train_augment,
    )

    # Infer number of classes from the dataset (works even if dataset contains 7 classes)
    num_classes = len(train_loader.dataset.classes)

    # Optionally replace train_loader with a balanced sampler to mitigate class imbalance
    if args.balanced_sampler:
        from torch.utils.data import WeightedRandomSampler

        # train_loader.dataset is an EmotionSubset exposing .targets
        targets = np.array(train_loader.dataset.targets)
        num_samples = len(targets)
        class_counts = np.bincount(targets, minlength=num_classes)
        # weight for each class: inverse frequency
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("Using WeightedRandomSampler for balanced training batches.")

    model = BaselineCNN(num_classes=num_classes, dropout_p=args.dropout_p).to(device)
    # Print total params for visibility and sanity-check
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BaselineCNN | Total params: {total_params:,}")

    # Class weights for imbalance using train_loader subset (can be disabled)
    class_counts = np.bincount(train_loader.dataset.targets, minlength=num_classes)
    if args.no_class_weights:
        class_weights = None
        print("Class weights disabled (--no_class_weights).")
    else:
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_classes
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    # Scheduler mode depends on monitor
    sched_mode = "max" if args.monitor == "val_acc" else "min"
    scheduler = ReduceLROnPlateau(optimizer, mode=sched_mode, factor=0.75, patience=5)

    # Setup AMP scaler if requested and CUDA available
    use_amp = args.use_amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # best monitoring value initialization
    if args.monitor == "val_loss":
        best_monitor = float("inf")
    else:
        best_monitor = -float("inf")
    best_epoch = -1
    patience_counter = 0

    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_baseline.pt")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=(scaler if use_amp else None)
        )
        print(
            f"  Train   | loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['acc']:.4f}, f1={train_metrics['f1']:.4f}"
        )

        val_metrics = eval_one_epoch(model, val_loader, criterion, device)
        print(
            f"  Val     | loss={val_metrics['loss']:.4f}, "
            f"acc={val_metrics['acc']:.4f}, f1={val_metrics['f1']:.4f}"
        )

        # Step scheduler with appropriate monitored metric
        to_step = val_metrics["acc"] if args.monitor == "val_acc" else val_metrics["loss"]
        scheduler.step(to_step)

        # record history for CSV and plotting
        if 'history' not in locals():
            history = {
                "epoch": [],
                "train_loss": [],
                "train_acc": [],
                "train_f1": [],
                "val_loss": [],
                "val_acc": [],
                "val_f1": [],
            }
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1"].append(val_metrics["f1"])

        # Determine improvement according to monitor
        current_monitor = val_metrics["loss"] if args.monitor == "val_loss" else val_metrics["acc"]
        improved = (current_monitor < best_monitor) if args.monitor == "val_loss" else (current_monitor > best_monitor)
        if improved:
            best_monitor = current_monitor
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  --> New best baseline saved to {best_ckpt_path}")
        else:
            patience_counter += 1
            print(f"  No improvement, patience={patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    monitor_name = args.monitor
    print(f"\nBaseline training finished. Best epoch: {best_epoch} with best {monitor_name}={best_monitor:.4f}")

    # Final evaluation
    print("Loading best baseline model for final test evaluation...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.to(device)

    test_metrics = eval_one_epoch(model, test_loader, criterion, device)
    print(
        f"Test    | loss={test_metrics['loss']:.4f}, "
        f"acc={test_metrics['acc']:.4f}, f1={test_metrics['f1']:.4f}"
    )

    # Save baseline training metrics to CSV in checkpoint dir
    metrics_csv = os.path.join(args.checkpoint_dir, "baseline_metrics.csv")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    try:
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"])
            for i in range(len(history["epoch"])):
                writer.writerow([
                    history["epoch"][i],
                    history["train_loss"][i],
                    history["train_acc"][i],
                    history["train_f1"][i],
                    history["val_loss"][i],
                    history["val_acc"][i],
                    history["val_f1"][i],
                ])
        print(f"Saved baseline metrics to {metrics_csv}")
    except Exception as e:
        print(f"Warning: failed to save baseline metrics CSV: {e}")

    # Plot baseline training curves
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

        plot_path = os.path.join(args.checkpoint_dir, "baseline_training_curves.png")
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved baseline training curves to {plot_path}")
    except Exception as e:
        print(f"Warning: failed to save baseline training plots: {e}")


if __name__ == "__main__":
    main()
