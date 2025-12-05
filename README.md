# Clash Royale Emote Predictor
Facial Emotion Recognition → Clash Royale Emote Recommendation (Real-Time Webcam App)

This project trains a deep learning model to recognize human facial expressions and map them to Clash Royale emotes in real time. Using the FER-2013 dataset and a fine-tuned ResNet, the system takes webcam input, predicts an emotion, and overlays the corresponding emote icon on the video feed.

This repository implements a complete, modular ML pipeline that satisfies strict reproducibility and grading requirements.

## 📂 Directory Structure

project_root/
│
├── src/
│   ├── train.py                  # Training loop
│   ├── eval.py                   # Evaluation script
│   ├── realtime_demo.py          # Live webcam → emote output
│   ├── emote_mapping.py          # Emotion→emote name mappings
│   ├── data/
│   │   ├── __init__.py
│   │   └── fer_dataset.py        # Folder-based FER-2013 dataset loader
│   └── models/
│       ├── __init__.py
│       └── resnet_emotion.py     # ResNet model wrapper
│
├── data/
│   ├── train/<emotion>/*.jpg     # FER-2013 train images
│   ├── test/<emotion>/*.jpg      # FER-2013 test images
│   └── emotes/<emotion>.<ext>    # Clash Royale emote images
│
├── models/
│   └── checkpoints/best_resnet.pt
│
├── notebooks/                    # EDA, training curves, hparam search
├── videos/                       # Demo & technical walkthrough videos
├── docs/                         # Report, figures, and extra documentation
│
├── requirements.txt
└── README.md
## 🎯 Project Goal

Build a real-time system that:

Detects a user’s facial expression from a webcam stream

Classifies the emotion (angry, disgust, fear, happy, neutral, sad)

Maps that emotion to a Clash Royale emote

Displays the emote (text + image) directly over the webcam feed

Example:

If the model predicts “happy”, the app displays the Laughing King emote.

## 🧠 Model Overview
Architecture

Pretrained ResNet-18 (or ResNet-50)

Final layer replaced with a 6-class classifier

Dropout for regularization

Trained with ImageNet normalization

Dataset

Folder-organized FER-2013, structured as:

data/train/happy/*.jpg
data/train/sad/*.jpg
...
data/test/<emotion>/*.jpg


Classes:

angry, disgust, fear, happy, neutral, sad


Augmentations include:

Random horizontal flip

Random rotation

Color jitter

Random resized cropping

## 🧪 Training

To train the model:

python -m src.train --train_dir data/train --test_dir data/test


Key features in the training pipeline:

GPU support

Early stopping

Class-weighted loss (handles imbalance)

Optimizer comparison (SGD vs AdamW)

LR scheduling with ReduceLROnPlateau

Saves best model to:

models/checkpoints/best_resnet.pt

## 📊 Evaluation

To evaluate the trained model:

python -m src.eval --train_dir data/train --test_dir data/test


This outputs:

Test accuracy

Macro-F1 score

Confusion matrix

Full classification report

## 🎥 Real-Time Demo (Webcam App)

The live emote predictor overlays:

The predicted emotion

The corresponding Clash Royale emote icon

Run it with:

python -m src.realtime_demo \
    --checkpoint_path models/checkpoints/best_resnet.pt


Make sure your emote images live here:

data/emotes/happy.png
data/emotes/angry.png
...


The window will open and update predictions every frame.
Press q to quit.

## 🗺️ Emotion → Emote Mapping

Handled in:

src/emote_mapping.py


Default mapping (customizable):

Emotion	Emote
angry	angry.png
disgust	disgust.png
fear	fear.png
happy	happy.png
neutral	neutral.png
sad	sad.png
## 🧩 Rubric Items Satisfied (High-Level)

✔ Modular code design
✔ Train/val/test splits
✔ Image augmentations
✔ Preprocessing + normalization
✔ Transfer learning
✔ Hyperparameter tuning
✔ Regularization (dropout + weight decay + early stopping)
✔ GPU acceleration
✔ LR scheduling
✔ Optimizer comparison
✔ Real-time inference system
✔ Evaluation metrics: accuracy, macro-F1, confusion matrix
✔ Trained on large (>10k) CV dataset

Full mapping is included in docs/REPORT.md.

## 🔧 Installation
pip install -r requirements.txt


Make sure you have:

Python 3.8+

PyTorch + Torchvision

OpenCV (opencv-python)

numpy, pandas, scikit-learn

## 📞 Contact & Contribution

Feel free to open an issue or request help if something breaks.
Contributions are welcome via pull request.

## ⭐ Acknowledgments

FER-2013 dataset

PyTorch & Torchvision

Clash Royale emote art (property of Supercell, used for academic purposes)