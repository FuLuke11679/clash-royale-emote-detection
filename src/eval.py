# src/eval.py

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.data.fer_dataset import get_dataloaders
from src.models.resnet_emotion import EmotionResNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FER-2013 Emotion ResNet (folder-based)")
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--test_dir", type=str, default="data/test")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/checkpoints/best_resnet.pt",
    )
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # We only need the test loader here, but get_dataloaders returns all three.
    _, _, test_loader = get_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
    )
    # Infer number of classes from the test dataset (keeps eval compatible with different folder sets)
    num_classes = len(test_loader.dataset.classes)
    model = EmotionResNet(
        backbone=args.backbone,
        num_classes=num_classes,
        pretrained=False,
    )
    state = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
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

    # Use the class names discovered by ImageFolder (ensures correct ordering)
    label_names = list(test_loader.dataset.classes)
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))


if __name__ == "__main__":
    main()
