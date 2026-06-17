"""Inference entry points for MNIST predictions."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from inference.utils import get_class_probabilities, load_model, preprocess_image, resolve_device


def predict_image(
	image_path: str | Path,
	checkpoint_path: str | Path = "project/training/checkpoints/best_model.pt",
	device: str | None = None,
) -> dict[str, object]:
	"""Predict the digit in a single handwritten image."""
	device = resolve_device(device)
	model = load_model(checkpoint_path, device=device)
	image_tensor = preprocess_image(image_path).to(device)
	with torch.no_grad():
		logits = model(image_tensor)
		prediction = torch.argmax(logits, dim=1).item()
		probabilities = get_class_probabilities(logits)
	return {"prediction": int(prediction), "probabilities": probabilities}


def annotate_prediction(image_path: str | Path, prediction: int, output_path: str | Path) -> None:
	"""Save a copy of the uploaded image with the predicted digit stamped on it."""
	image = Image.open(image_path).convert("RGB")
	draw = ImageDraw.Draw(image)
	draw.rectangle((0, 0, 140, 32), fill="black")
	draw.text((8, 8), f"Predicted: {prediction}", fill="white")
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	image.save(output_path)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Predict a handwritten digit from an image file.")
	parser.add_argument("image", help="Path to the handwritten digit image")
	parser.add_argument("--checkpoint", default="project/training/checkpoints/best_model.pt")
	args = parser.parse_args()

	result = predict_image(args.image, checkpoint_path=args.checkpoint)
	print(f"Predicted digit: {result['prediction']}")
	print("Class probabilities:")
	for digit, probability in enumerate(result["probabilities"]):
		print(f"  {digit}: {probability:.4f}")
