# SETUP

This file contains quick setup instructions for the Clash Royale Emote Predictor project.

Prerequisites
- Python 3.8+
- pip
- (Optional) CUDA-enabled GPU and appropriate PyTorch build

Install dependencies
```bash
pip install -r requirements.txt
```

GPU notes
- If you have a CUDA-capable GPU, install a matching `torch`/`torchvision` wheel from https://pytorch.org before running heavy training.
- Ensure CUDA drivers are installed and `torch.cuda.is_available()` returns True.

Run commands
- Train: `python -m src.train --train_dir data/train --test_dir data/test --checkpoint_dir models/checkpoints`
- Evaluate: `python -m src.eval --test_dir data/test --checkpoint_path models/checkpoints/best_resnet.pt`
- Demo: `python -m src.realtime_demo --checkpoint_path models/checkpoints/best_resnet.pt --emote_assets_dir data/emotes`

Troubleshooting
- If images are grayscale, the dataset loader will convert them to RGB automatically; if you see channel mismatch errors, check `src/data/fer_dataset.py`.
