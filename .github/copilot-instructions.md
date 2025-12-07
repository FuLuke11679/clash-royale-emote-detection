# 🤖 Copilot / AI Agent Instructions
Project: Clash Royale Emote Predictor (FER-2013 → Real-Time Emote Webcam System)
MANDATORY: All edits MUST preserve rubric compliance.

If a requested change would violate the rubric, suggest an alternative instead.

----------------------------------------
🎓 RUBRIC REQUIREMENTS (MUST NOT BE BROKEN)
🎯 1. Machine Learning Requirements (Max 15 Items)

This project MUST include the following ML features.
Any edits must keep these intact:

Core ML Fundamentals

Modular code design (multiple modules under src/)

Explicit train/validation/test split

Training curve visualization (loss + accuracy)

Baseline model (simple CNN/MLP for comparison)

Regularization: dropout, weight decay, early stopping

Hyperparameter tuning (LR, batch size, optimizer)

Data augmentation: ≥4 transforms (flip, rotation, jitter, crop)

Data Preprocessing

Image normalization and standardization

Preprocessing pipeline handling data quality issues
(resizing, grayscale→RGB where needed, class imbalance)

Optimization

Learning rate scheduling (ReduceLROnPlateau)

GPU acceleration

Optimizer comparison (SGD vs. AdamW)

Transfer Learning

Fine-tuned pretrained ResNet (ImageNet initialization)

Computer Vision

CNN-based model

Large dataset processing (>10K images → FER-2013)

Advanced Systems

Real-time inference with latency constraints (webcam app)

Copilot must NOT remove, weaken, or break any of these items.

🎯 2. Following Directions Requirements (30 pts)

The repo MUST include these structure and documentation rules:

Project Directory Structure

Copilot must maintain:

src/
models/
data/
notebooks/
videos/
docs/
requirements.txt (or environment.yml)

Two Required Videos

Demo Video (non-technical)

Technical Walkthrough Video

These belong in videos/.

Required Documentation

README.md

SETUP.md

ATTRIBUTION.md

Self-Assessment (user will submit separately but code must support it)

Do NOT break runnable commands

All terminal invocations must continue to work:

python -m src.train ...
python -m src.eval ...
python -m src.realtime_demo ...

🎯 3. Project Cohesion Requirements (20 pts)

All contributions MUST maintain:

A unified narrative: FER-2013 → ResNet → real-time emote inference

Clean, modular code that matches the README

Directory consistency

Aligned documentation / structure / naming

Evaluation that matches project goals (emotion classification + demo)

Do not introduce unrelated ML tasks, datasets, or architectures.

----------------------------------------
🧭 Overall Project Purpose (High-Level)

Train a transfer-learned CNN on the folder-structured FER-2013 dataset, then run a real-time webcam inference system that overlays the matching Clash Royale emote image based on predicted emotion.

Never change this purpose.

----------------------------------------
📐 Critical Architecture Summary (Do Not Alter)
data/train/<emotion>/*.jpg
data/test/<emotion>/*.jpg
      ↓
src/data/fer_dataset.py → ImageFolder + val split + transforms
      ↓
src/models/resnet_emotion.py → pretrained ResNet18/50 (6 classes)
      ↓
src/train.py → training loop, LR scheduler, class weights, augmentation
      ↓
models/checkpoints/best_resnet.pt
      ↓
src/eval.py → accuracy, macro-F1, confusion matrix
      ↓
src/realtime_demo.py → webcam inference + emote overlay


Also uses:

data/emotes/<emotion>.png

----------------------------------------
📂 Key Files to Preserve Behavior For
src/data/fer_dataset.py

ImageFolder-based FER-2013 loader

Stratified train/val split

Train vs val/test transforms

ImageNet normalization

If transforms change → update realtime demo too

src/models/resnet_emotion.py

Pretrained ResNet wrapper

FC replacement for 6-class head

Dropout support

Backwards-compatible checkpoint loading

Torchvision weight API compatibility

src/train.py

Optimizers: AdamW + SGD

Scheduler: ReduceLROnPlateau

Early stopping

Weighted loss for imbalance

Best-model saving

GPU support

Augmentation logic

src/eval.py

Computes accuracy + macro-F1

Shows confusion matrix

Mapping via IDX_TO_EMOTION

src/realtime_demo.py

Webcam → transform → inference → emote overlay

Low latency required

Emote filenames from data/emotes/<emotion>.<ext>

Uses same transforms as eval

src/emote_mapping.py

Must stay in sync with:

FER class folder names

Emote filenames

----------------------------------------
🧪 Essential Developer Workflows (Must Stay Valid)
Install
pip install -r requirements.txt

Train
python -m src.train --train_dir data/train --test_dir data/test

Evaluate
python -m src.eval --test_dir data/test --checkpoint_path models/checkpoints/best_resnet.pt

Real-time Demo
python -m src.realtime_demo --checkpoint_path models/checkpoints/best_resnet.pt --emote_assets_dir data/emotes

----------------------------------------
⚠️ Project-Specific Gotchas (AI Must Honor These)
✔ Class ordering MUST match folder names

Defined in:

CLASS_NAMES = ["angry","disgust","fear","happy","neutral","sad", "surprise"]

✔ Any preprocessing changes must propagate to realtime demo

Otherwise demo predictions become wrong.

✔ Do not remove early stopping or scheduler

Rubric requires both.

✔ Loss function uses class weights

Preserve imbalance handling.

✔ Checkpoint loading must remain compatible

Changing model structure risks breaking saved models.

✔ NEVER slow down real-time inference

No heavy transforms inside the inference loop.

----------------------------------------
🛠️ Safe Editing Policies for Copilot

Use package-relative imports: from src.data...

Maintain the directory structure

Keep ML logic modular (no monolithic scripts)

Preserve all ML rubric items already implemented

If asked to implement a feature that conflicts with the rubric:

Warn the user

Provide a compliant alternative

----------------------------------------
🧪 Verification Steps (AI Should Use For Safety)

After editing, Copilot should internally check:

Model loads successfully
python -m src.eval --checkpoint_path models/checkpoints/best_resnet.pt

Real-time demo works
python -m src.realtime_demo

Train script runs on a subset

(with no crash)

Transforms are consistent across train/eval/demo
Class mappings match emotes and dataset folders
----------------------------------------
📬 If Instructions Need Updating

AI should ask:

“Which rubric category or pipeline behavior should the updated instructions reflect?”

## Agent quick reference (practical)

Quick architecture
- Dataset (folder layout) → `src/data/fer_dataset.py` (ImageFolder + stratified val split) → `src/models/resnet_emotion.py` (ResNet wrapper) → `src/train.py` (training loop, scheduler, early stopping) → checkpoints `models/checkpoints/best_resnet.pt` → `src/realtime_demo.py` for inference and overlay.

Key files
- `src/data/fer_dataset.py` — transforms, CLASS_NAMES, dataloaders
- `src/models/resnet_emotion.py` — model, `_unwrap_state_dict` compatibility helper
- `src/train.py` — use `--optimizer` (adamw|sgd), `--val_ratio`, `--checkpoint_dir`
- `src/eval.py` — smoke-check model loading and metrics
- `src/realtime_demo.py` — uses `EMOTION_TO_EMOTE` and loads images from `data/emotes/`

Concrete commands (copy-paste)
```
pip install -r requirements.txt
python -m src.train --train_dir data/train --test_dir data/test --checkpoint_dir models/checkpoints
python -m src.eval --test_dir data/test --checkpoint_path models/checkpoints/best_resnet.pt
python -m src.realtime_demo --checkpoint_path models/checkpoints/best_resnet.pt --emote_assets_dir data/emotes
```

Quick gotchas
- Keep `CLASS_NAMES` in `src/data/fer_dataset.py` in sync with folder names and `src/emote_mapping.py`.
- Any change to `get_transforms` must be mirrored in `realtime_demo.build_preprocess_transform()`.
- Checkpoint loads may be wrapped or contain `module.` prefixes; follow `_unwrap_state_dict` behavior when updating model shape.
- Avoid heavy transforms inside the live inference loop to preserve latency.

Fast local smoke test
- Train a tiny run to validate wiring (low-cost):
```
python -m src.train --train_dir data/train --test_dir data/test --epochs 1 --batch_size 8 --num_workers 0 --checkpoint_dir models/checkpoints
python -m src.eval --test_dir data/test --checkpoint_path models/checkpoints/best_resnet.pt
```

Verification after edits
- Run `python -m src.eval --checkpoint_path models/checkpoints/best_resnet.pt` to confirm model loads.
- Run `python -m src.realtime_demo` (with `--checkpoint_path`) to verify the demo still shows emote overlays and runs at low latency.

If you want this quick reference split into a separate file instead, say so and I will create `.github/copilot-quickstart.md` instead of appending.