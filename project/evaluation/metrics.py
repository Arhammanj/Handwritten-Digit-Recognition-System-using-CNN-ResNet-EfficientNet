"""Metrics for MNIST classification."""

from __future__ import annotations

import torch
from sklearn.metrics import confusion_matrix


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> int:
	"""Count correct predictions from raw model outputs."""
	predictions = torch.argmax(logits, dim=1)
	return int((predictions == labels).sum().item())


def accuracy_score(num_correct: int, num_total: int) -> float:
	"""Compute accuracy as a fraction."""
	if num_total == 0:
		return 0.0
	return num_correct / num_total


def collect_predictions(model, data_loader, device: torch.device):
	"""Collect labels and predictions for confusion-matrix plotting."""
	model.eval()
	all_true = []
	all_pred = []
	with torch.no_grad():
		for images, labels in data_loader:
			images = images.to(device)
			logits = model(images)
			predictions = torch.argmax(logits, dim=1).cpu().tolist()
			all_pred.extend(predictions)
			all_true.extend(labels.tolist())
	return all_true, all_pred


def build_confusion_matrix(true_labels, pred_labels):
	"""Return a 10x10 confusion matrix for MNIST digits."""
	return confusion_matrix(true_labels, pred_labels, labels=list(range(10)))
