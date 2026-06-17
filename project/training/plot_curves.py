"""Standalone plotting script for MNIST training curves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from training.train import load_history


def plot_from_history(history_path: str | Path, output_path: str | Path) -> None:
	history = load_history(history_path)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	epochs = range(1, len(history["train_loss"]) + 1)
	fig, axes = plt.subplots(1, 2, figsize=(12, 4))

	axes[0].plot(epochs, history["train_loss"], label="Train Loss")
	axes[0].plot(epochs, history["val_loss"], label="Val Loss")
	axes[0].set_title("Loss vs Epoch")
	axes[0].set_xlabel("Epoch")
	axes[0].set_ylabel("Loss")
	axes[0].legend()
	axes[0].grid(True, alpha=0.3)

	axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
	axes[1].plot(epochs, history["val_acc"], label="Val Accuracy")
	axes[1].set_title("Accuracy vs Epoch")
	axes[1].set_xlabel("Epoch")
	axes[1].set_ylabel("Accuracy")
	axes[1].legend()
	axes[1].grid(True, alpha=0.3)

	fig.tight_layout()
	fig.savefig(output_path, dpi=200)
	plt.close(fig)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plot MNIST training curves from saved history.")
	parser.add_argument("--history", default="project/training/training_history.json")
	parser.add_argument("--output", default="project/training/learning_curves.png")
	args = parser.parse_args()

	plot_from_history(args.history, args.output)
	print(f"Saved plot to {args.output}")