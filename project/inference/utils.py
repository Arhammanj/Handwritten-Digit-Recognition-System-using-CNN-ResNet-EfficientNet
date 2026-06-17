"""Utilities for loading a trained MNIST model and preparing images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model.cnn import DigitCNN


def resolve_device(device: str | None = None) -> torch.device:
	if device is not None:
		return torch.device(device)
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str | Path, device: str | None = None) -> DigitCNN:
	"""Load a trained CNN from disk."""
	device = resolve_device(device)
	model = DigitCNN().to(device)
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	return model


def preprocess_image(image_path: str | Path) -> torch.Tensor:
	"""Convert a handwritten digit image into a normalized tensor."""
	image = Image.open(image_path).convert("L").resize((28, 28))
	array = np.asarray(image, dtype=np.float32) / 255.0
	array = 1.0 - array
	return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def get_class_probabilities(logits: torch.Tensor) -> list[float]:
	"""Convert model logits into readable digit probabilities."""
	probabilities = torch.softmax(logits, dim=1).squeeze(0)
	return probabilities.detach().cpu().tolist()
