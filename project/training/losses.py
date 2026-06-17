"""Loss helpers for MNIST training."""

from __future__ import annotations

import torch.nn as nn


def get_loss_function() -> nn.Module:
	"""Return the standard classification loss for digit recognition."""
	return nn.CrossEntropyLoss()
