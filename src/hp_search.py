"""Simple hyperparameter search harness for baseline model.

Runs a small set of configurations (short epochs) and records final validation/test metrics.

This script uses the command-line training script (`python -m src.train_baseline`) so it's
lightweight and doesn't duplicate training code. It's designed for quick local experiments
and produces `models/checkpoints/hp_search_summary.csv` and a small plot.

Usage:
    python -m src.hp_search --out_dir models/checkpoints
"""
import argparse
import itertools
import os
import shutil
import subprocess
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="models/checkpoints")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def run_config(idx, cfg, out_dir, epochs, batch_size, num_workers):
    run_dir = os.path.join(out_dir, f"hp_run_{idx}")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    args = [
        "python", "-m", "src.train_baseline",
        "--checkpoint_dir", run_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
    ]

    # include optimizer and lr if provided in cfg
    if "optimizer" in cfg:
        args.extend(["--optimizer", str(cfg["optimizer"])])
    if "lr" in cfg:
        args.extend(["--lr", str(cfg["lr"])])

    if cfg.get("balanced_sampler"):
        args.append("--balanced_sampler")
    if cfg.get("no_class_weights"):
        args.append("--no_class_weights")
    # tell trainer whether to use augmentation via env var
    env = os.environ.copy()
    if not cfg.get("train_augment", True):
        env["TRAIN_AUGMENT"] = "0"

    print("Running config:", cfg)
    start = time.time()
    subprocess.run(args, check=True, env=env)
    duration = time.time() - start

    # After training, run eval_baseline to produce a CSV
    eval_csv = os.path.join(run_dir, "hp_eval_summary.csv")
    eval_args = [
        "python", "-m", "src.eval_baseline",
        "--checkpoint_path", os.path.join(run_dir, "best_baseline.pt"),
        "--output_csv", eval_csv,
    ]
    subprocess.run(eval_args, check=True)

    # Read CSV
    metrics = None
    if os.path.exists(eval_csv):
        with open(eval_csv, "r") as f:
            # two-line CSV: header + values
            _ = f.readline()
            vals = f.readline().strip().split(',')
            metrics = {"loss": float(vals[0]), "accuracy": float(vals[1]), "macro_f1": float(vals[2])}

    return {**cfg, **(metrics or {}), "run_dir": run_dir, "duration": duration}


def main():
    args = parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Small grid (kept intentionally tiny for local smoke runs)
    lrs = [1e-3, 5e-4]
    opts = ["adamw", "sgd"]
    samplers = [False, True]
    no_weights = [False, True]
    aug = [True, False]

    combos = []
    for lr, opt, samp, nw, tr_aug in itertools.islice(itertools.product(lrs, opts, samplers, no_weights, aug), 8):
        combos.append({"lr": lr, "optimizer": opt, "balanced_sampler": samp, "no_class_weights": nw, "train_augment": tr_aug})

    results = []
    for i, cfg in enumerate(combos):
        try:
            r = run_config(i, cfg, out_dir, epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers)
            results.append(r)
        except subprocess.CalledProcessError as e:
            print(f"Run {i} failed: {e}")

    # Write summary CSV
    summary_csv = os.path.join(out_dir, "hp_search_summary.csv")
    keys = ["run_dir", "optimizer", "lr", "balanced_sampler", "no_class_weights", "train_augment", "loss", "accuracy", "macro_f1", "duration"]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for r in results:
            writer.writerow([r.get(k, "") for k in keys])

    # Plot accuracy comparison
    try:
        plt.figure(figsize=(8, 4))
        accs = [r.get("accuracy", 0) for r in results]
        labels = [f"{r['optimizer']}_lr{r['lr']}_s{int(r['balanced_sampler'])}_w{int(r['no_class_weights'])}_a{int(r['train_augment'])}" for r in results]
        plt.bar(range(len(accs)), accs)
        plt.xticks(range(len(accs)), labels, rotation=45, ha="right")
        plt.ylabel("Test accuracy")
        plt.tight_layout()
        plot_path = os.path.join(out_dir, "hp_search_accuracy.png")
        plt.savefig(plot_path)
        print(f"Saved hp search plot to {plot_path}")
    except Exception as e:
        print("Failed to plot hp search results:", e)

    print(f"Wrote summary to {summary_csv}")


if __name__ == "__main__":
    main()
