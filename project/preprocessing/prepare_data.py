"""Load MNIST-style raw data from the project raw folder.

The raw upload in this workspace is stored as IDX files inside
``project/dataset/raw/archive (6)``. This script discovers those files,
reads them into memory, and exposes a small PyTorch dataset/dataloader
pipeline for quick experimentation.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "dataset" / "raw"
__all__ = [
	"MNISTDataset",
	"build_dataloaders",
	"load_mnist_split",
	"RAW_ROOT",
]


def _find_first_match(root: Path, filename: str) -> Path:
	"""Find the first file matching ``filename`` anywhere under ``root``."""
	matches = list(root.rglob(filename))
	if not matches:
		raise FileNotFoundError(f"Could not find {filename!r} under {root}")
	return matches[0]


def _read_idx_images(path: Path) -> np.ndarray:
	"""Read an IDX image file into a ``(N, H, W)`` NumPy array."""
	with path.open("rb") as file_handle:
		magic, num_images, num_rows, num_cols = struct.unpack(">IIII", file_handle.read(16))
		if magic != 2051:
			raise ValueError(f"Unexpected IDX image magic number {magic} in {path}")
		data = np.frombuffer(file_handle.read(), dtype=np.uint8)
	return data.reshape(num_images, num_rows, num_cols)


def _read_idx_labels(path: Path) -> np.ndarray:
	"""Read an IDX label file into a 1D NumPy array."""
	with path.open("rb") as file_handle:
		magic, num_labels = struct.unpack(">II", file_handle.read(8))
		if magic != 2049:
			raise ValueError(f"Unexpected IDX label magic number {magic} in {path}")
		labels = np.frombuffer(file_handle.read(), dtype=np.uint8)
	return labels


def load_mnist_split(split_name: str) -> tuple[np.ndarray, np.ndarray]:
	"""Load one MNIST split (``train`` or ``t10k``) from the raw folder."""
	images_path = _find_first_match(RAW_ROOT, f"{split_name}-images.idx3-ubyte")
	labels_path = _find_first_match(RAW_ROOT, f"{split_name}-labels.idx1-ubyte")
	images = _read_idx_images(images_path)
	labels = _read_idx_labels(labels_path)
	return images, labels


class MNISTDataset(Dataset):
	"""Simple PyTorch dataset for grayscale MNIST images."""

	def __init__(self, images: np.ndarray, labels: np.ndarray):
		self.images = images
		self.labels = labels

	def __len__(self) -> int:
		return len(self.images)

	def __getitem__(self, index: int):
		image = torch.tensor(self.images[index], dtype=torch.float32).unsqueeze(0) / 255.0
		label = torch.tensor(self.labels[index], dtype=torch.long)
		return image, label


def build_dataloaders(batch_size: int = 64, val_size: float = 0.1, seed: int = 42):
	"""Create train/validation/test dataloaders from the raw MNIST files."""
	train_images, train_labels = load_mnist_split("train")
	test_images, test_labels = load_mnist_split("t10k")

	train_indices, val_indices = train_test_split(
		np.arange(len(train_images)), test_size=val_size, random_state=seed, stratify=train_labels
	)

	train_dataset = MNISTDataset(train_images, train_labels)
	val_dataset = MNISTDataset(train_images, train_labels)
	test_dataset = MNISTDataset(test_images, test_labels)

	train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, test_loader


if __name__ == "__main__":
	print(f"RAW_ROOT: {RAW_ROOT}")
	train_loader, val_loader, test_loader = build_dataloaders()

	images, labels = next(iter(train_loader))
	print(f"Train batch images: {images.shape}")
	print(f"Train batch labels: {labels.shape}")
	print(f"Validation batches: {len(val_loader)}")
	print(f"Test batches: {len(test_loader)}")