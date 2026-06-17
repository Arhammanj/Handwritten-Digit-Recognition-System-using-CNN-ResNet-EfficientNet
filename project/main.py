"""Project entry point for training, evaluation, and inference."""

from __future__ import annotations

import argparse

from evaluation.evaluate import evaluate_model
from inference.predict import predict_image
from evaluation.evaluate import plot_confusion_matrix
from preprocessing.prepare_data import build_dataloaders
from training.train import load_checkpoint, train_model
from model.cnn import build_model


def main() -> None:
	parser = argparse.ArgumentParser(description="MNIST CNN project")
	parser.add_argument("--mode", choices=["train", "evaluate", "predict"], default="train")
	parser.add_argument("--image", help="Path to an image for prediction")
	parser.add_argument("--checkpoint", default="project/training/checkpoints/best_model.pt")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--model", choices=["cnn", "resnet", "efficientnet"], default="cnn")
	args = parser.parse_args()

	train_loader, val_loader, test_loader = build_dataloaders()

	if args.mode == "train":
		model, _history = train_model(
			train_loader,
			val_loader,
			epochs=args.epochs,
			model_name=args.model,
			checkpoint_path=args.checkpoint,
		)
		test_metrics = evaluate_model(model, test_loader)
		print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
		plot_confusion_matrix(model, test_loader, "project/evaluation/confusion_matrix.png")
		print("Confusion matrix saved to project/evaluation/confusion_matrix.png")

	elif args.mode == "evaluate":
		model = build_model(args.model)
		model = load_checkpoint(model, args.checkpoint)
		test_metrics = evaluate_model(model, test_loader)
		print(f"Test loss: {test_metrics['loss']:.4f}")
		print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
		plot_confusion_matrix(model, test_loader, "project/evaluation/confusion_matrix.png")
		print("Confusion matrix saved to project/evaluation/confusion_matrix.png")

	elif args.mode == "predict":
		if not args.image:
			raise ValueError("--image is required when --mode predict is used")
		result = predict_image(args.image, checkpoint_path=args.checkpoint)
		print(f"Predicted digit: {result['prediction']}")
		print("Class probabilities:")
		for digit, probability in enumerate(result["probabilities"]):
			print(f"  {digit}: {probability:.4f}")


if __name__ == "__main__":
	main()