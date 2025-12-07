import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """SmallVGG-like baseline adapted for this repo.

    - Accepts configurable `in_channels` (default 3) so it works with RGB inputs
    - Uses adaptive pooling to keep the classifier size stable
    """

    def __init__(self, num_classes: int = 7, dropout_p: float = 0.5, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(256),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            nn.Conv2d(256,512,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.Conv2d(512,512,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((2,2)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*2*2, 1024), nn.ReLU(True), nn.Dropout(dropout_p),
            nn.Linear(1024, 512),    nn.ReLU(True), nn.Dropout(dropout_p),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
