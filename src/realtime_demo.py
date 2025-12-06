# src/realtime_demo.py

import argparse
import os
from typing import Tuple, Dict

import cv2
import numpy as np
import torch
from torchvision import transforms

from models.resnet_emotion import EmotionResNet
from emote_mapping import IDX_TO_EMOTION, EMOTION_TO_EMOTE
from data.fer_dataset import NUM_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time FER -> Clash emote demo")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/checkpoints/best_resnet.pt",
    )
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument(
        "--emote_assets_dir",
        type=str,
        default="data/emotes",  # <- your emotes folder
    )
    return parser.parse_args()


def build_preprocess_transform():
    """
    Mirror the val/test transforms from fer_dataset.py:
    Resize -> CenterCrop -> ToTensor -> Normalize
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.ToTensor(),  # HWC uint8 -> CHW float [0,1]
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def load_emote_images(assets_dir: str) -> Dict[str, np.ndarray]:
    """
    Load emote images from disk into memory.

    Expects filenames like: angry.png, happy.jpg, etc.
    Keys in returned dict are the base names without extension,
    e.g. 'angry', 'happy'.
    """
    emote_images = {}
    if not os.path.isdir(assets_dir):
        print(f"[WARN] Emote assets dir not found: {assets_dir}. Using text only.")
        return emote_images

    for fname in os.listdir(assets_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        emote_name = os.path.splitext(fname)[0]  # 'angry' from 'angry.png'
        img_path = os.path.join(assets_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # keep alpha if present
        if img is not None:
            emote_images[emote_name] = img
    print(f"Loaded {len(emote_images)} emote images from {assets_dir}")
    return emote_images


def overlay_emote_image(frame: np.ndarray, emote_img: np.ndarray, x: int = 10, y: int = 10) -> np.ndarray:
    """
    Overlay emote_img onto frame at (x, y).
    Handles alpha channel if emote_img has it.
    """
    if emote_img is None:
        return frame

    h, w = emote_img.shape[:2]

    # Ensure it fits in frame
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        h = min(h, frame.shape[0] - y)
        w = min(w, frame.shape[1] - x)
        emote_img = emote_img[0:h, 0:w]

    if emote_img.shape[2] == 4:
        # BGRA image with alpha channel
        alpha = emote_img[:, :, 3] / 255.0
        alpha = alpha[..., None]  # (h, w, 1)
        fg = emote_img[:, :, :3]
        bg = frame[y : y + h, x : x + w]
        blended = alpha * fg + (1 - alpha) * bg
        frame[y : y + h, x : x + w] = blended.astype(np.uint8)
    else:
        frame[y : y + h, x : x + w] = emote_img

    return frame


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = EmotionResNet(
        backbone=args.backbone,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )
    state = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    preprocess = build_preprocess_transform()

    # Load emote images (from data/emotes)
    emote_images = load_emote_images(args.emote_assets_dir)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {args.camera_index}")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        # Convert BGR -> RGB for transforms
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess
        img_tensor = preprocess(rgb_frame)  # (3, 224, 224)
        img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)

        with torch.inference_mode():
            logits = model(img_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        pred_idx = int(preds[0])
        emotion = IDX_TO_EMOTION[pred_idx]
        emote_key = EMOTION_TO_EMOTE.get(emotion, emotion)

        # Overlay text
        text = f"Emotion: {emotion.upper()} | Emote: {emote_key}"
        cv2.putText(
            frame,
            text,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Overlay emote image, if we have it
        emote_img = emote_images.get(emote_key, None)
        if emote_img is not None:
            frame = overlay_emote_image(frame, emote_img, x=10, y=10)

        cv2.imshow("Clash Royale Emote Recommender", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
