# src/models/resnet_emotion.py

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict


class EmotionResNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        if backbone == "resnet18":
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet18(weights=weights)
            except Exception:
                # fallback for older torchvision versions
                self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
        elif backbone == "resnet50":
            try:
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet50(weights=weights)
            except Exception:
                self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace last FC layer with Dropout + Linear
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _unwrap_state_dict(state_dict: dict) -> dict:
    """Handle checkpoints saved as {'state_dict':..., 'model_state_dict':..., or raw state_dict} and strip 'module.' prefixes."""
    if not isinstance(state_dict, dict):
        return state_dict
    # common wrappers
    for key in ("state_dict", "model_state_dict", "model", "net"):
        if key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
            break

    new_state = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v
    return new_state


def load_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    backbone: str = "resnet18",
    num_classes: int = 7,
) -> EmotionResNet:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionResNet(backbone=backbone, num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = _unwrap_state_dict(state)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # try non-strict load (useful when checkpoint has mismatched keys)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model