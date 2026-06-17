"""Model evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from evaluation.metrics import accuracy_from_logits, accuracy_score, build_confusion_matrix, collect_predictions
from training.losses import get_loss_function


def evaluate_model(model, data_loader, device: str | None = None) -> dict[str, float]:
	"""Evaluate a trained model on a dataloader."""
	device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
	criterion = get_loss_function()
	model = model.to(device)
	model.eval()

	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	with torch.no_grad():
		for images, labels in data_loader:
			images = images.to(device)
			labels = labels.to(device)
			logits = model(images)
			loss = criterion(logits, labels)

			total_loss += loss.item() * labels.size(0)
			total_correct += accuracy_from_logits(logits, labels)
			total_samples += labels.size(0)

	return {
		"loss": total_loss / total_samples if total_samples else 0.0,
		"accuracy": accuracy_score(total_correct, total_samples),
	}


def plot_confusion_matrix(model, data_loader, plot_path: str | Path, device: str | None = None) -> None:
	"""Save a confusion matrix image for your report."""
	device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
	model = model.to(device)
	true_labels, pred_labels = collect_predictions(model, data_loader, device)
	cm = build_confusion_matrix(true_labels, pred_labels)

	plot_path = Path(plot_path)
	plot_path.parent.mkdir(parents=True, exist_ok=True)

	fig, ax = plt.subplots(figsize=(7, 7))
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
	disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
	ax.set_title("MNIST Confusion Matrix")
	fig.tight_layout()
	fig.savefig(plot_path, dpi=200)
	plt.close(fig)
