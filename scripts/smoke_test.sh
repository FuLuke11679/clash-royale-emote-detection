#!/usr/bin/env bash
# Quick smoke test: 1-epoch baseline train and eval (small batch)
set -e

echo "Running smoke test: baseline 1-epoch training"

# Choose dataset paths: prefer data/train & data/test, else fall back to data/face/train & data/face/test
if [ -d "data/train" ] && [ -d "data/test" ]; then
	TRAIN_DIR="data/train"
	TEST_DIR="data/test"
elif [ -d "data/face/train" ] && [ -d "data/face/test" ]; then
	TRAIN_DIR="data/face/train"
	TEST_DIR="data/face/test"
	echo "Note: using dataset under data/face/"
else
	echo "ERROR: could not find dataset folders. Expected either data/train & data/test or data/face/train & data/face/test"
	exit 1
fi

python -m src.train_baseline --train_dir "$TRAIN_DIR" --test_dir "$TEST_DIR" --epochs 1 --batch_size 8 --num_workers 0 --checkpoint_dir models/checkpoints

echo "Evaluating baseline checkpoint"
# The baseline script already prints final test metrics. Running the ResNet evaluator against
# the baseline checkpoint will fail (different model architecture). Only run `src.eval` if
# a ResNet checkpoint exists.
if [ -f models/checkpoints/best_resnet.pt ]; then
	echo "Found ResNet checkpoint — running resnet eval"
	python -m src.eval --train_dir "$TRAIN_DIR" --test_dir "$TEST_DIR" --checkpoint_path models/checkpoints/best_resnet.pt --batch_size 8 --num_workers 0
else
	echo "No ResNet checkpoint found (models/checkpoints/best_resnet.pt). Skipping ResNet evaluation."
fi

echo "Smoke test complete. Check models/checkpoints for artifacts (best_baseline.pt, baseline_metrics.csv, baseline_training_curves.png)."
