"""Evaluate the BaselineCNN checkpoint on the folder-structured test dataset.

Usage:
    python -m src.eval_baseline --test_dir data/face/test --checkpoint_path models/checkpoints/best_baseline.pt
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.data.fer_dataset import get_dataloaders
from src.models.baseline_cnn import BaselineCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline CNN on FER-2013 (folder-based)")
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--test_dir", type=str, default="data/test")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/checkpoints/best_baseline.pt",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--output_csv", type=str, default=None,
                        help="If set, write a one-line CSV with loss,acc,f1 to this path")
    return parser.parse_args()


def _unwrap_state_dict(state):
    # handle wrapper dicts
    if not isinstance(state, dict):
        return state
    for key in ("state_dict", "model_state_dict", "model", "net"):
        if key in state and isinstance(state[key], dict):
            return state[key]
    return state


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve dataset paths: accept either data/train + data/test or data/face/train + data/face/test
    tried = []
    train_dir = args.train_dir
    test_dir = args.test_dir
    if not os.path.isdir(train_dir):
        alt = os.path.join("data", "face", "train")
        if os.path.isdir(alt):
            print(f"Note: using dataset under {alt}")
            train_dir = alt
        else:
            tried.append(train_dir)
    if not os.path.isdir(test_dir):
        alt = os.path.join("data", "face", "test")
        if os.path.isdir(alt):
            print(f"Note: using dataset under {alt}")
            test_dir = alt
        else:
            tried.append(test_dir)

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        msg = (
            "Could not find dataset directories. Tried:\n"
            f" - {args.train_dir}\n"
            f" - data/face/train\n"
            f" - {args.test_dir}\n"
            f" - data/face/test\n"
            "Please set --train_dir and --test_dir to valid ImageFolder-style folders.")
        raise FileNotFoundError(msg)

    # get loaders (we only need test loader)
    _, _, test_loader = get_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
    )

    num_classes = len(test_loader.dataset.classes)
    model = BaselineCNN(num_classes=num_classes)

    # load checkpoint robustly
    state = torch.load(args.checkpoint_path, map_location=device)
    state_dict = _unwrap_state_dict(state)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []
    criterion = nn.CrossEntropyLoss()
    losses = []

    with torch.inference_mode():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = float(np.mean(losses))
    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average="macro"))
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test loss: {avg_loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test macro-F1: {f1:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    label_names = list(test_loader.dataset.classes)
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))

    # Diagnostics: logits statistics and collapse warning
    if len(all_logits) > 0:
        logits_arr = np.vstack(all_logits)
        per_class_std = logits_arr.std(axis=0)
        overall_std = float(per_class_std.mean())
        unique_preds = np.unique(all_preds)
        print(f"\nLogits per-class std (mean): {overall_std:.6f}")
        if overall_std == 0.0 or unique_preds.size == 1:
            print("WARNING: model outputs have zero variance or only a single predicted class. This often indicates a collapsed model or training issue.")

    # Optionally write a compact CSV for programmatic use
    if args.output_csv:
        import csv
        out_path = args.output_csv
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["loss", "accuracy", "macro_f1"])
            writer.writerow([f"{avg_loss:.6f}", f"{acc:.6f}", f"{f1:.6f}"])
        print(f"Wrote summary metrics to {out_path}")


if __name__ == "__main__":
    main()
