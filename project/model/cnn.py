"""CNN model for MNIST digit classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


class DigitCNN(nn.Module):
    """A compact CNN that learns strokes, edges, and digit shapes."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class DigitResNet(nn.Module):
    """ResNet18 adapted for grayscale MNIST digits."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class DigitEfficientNet(nn.Module):
    """EfficientNet-B0 adapted for grayscale MNIST digits."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def build_model(model_name: str = "cnn", num_classes: int = 10) -> nn.Module:
    """Factory for comparing CNN, ResNet, and EfficientNet on MNIST."""
    model_name = model_name.lower()
    if model_name == "cnn":
        return DigitCNN(num_classes=num_classes)
    if model_name == "resnet":
        return DigitResNet(num_classes=num_classes)
    if model_name == "efficientnet":
        return DigitEfficientNet(num_classes=num_classes)
    raise ValueError(f"Unknown model_name: {model_name}")