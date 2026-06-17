"""Training loop for the MNIST CNN."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from evaluation.evaluate import evaluate_model
from evaluation.metrics import accuracy_from_logits
from model.cnn import build_model
from training.losses import get_loss_function


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "training" / "checkpoints" / "best_model.pt"
DEFAULT_FINAL_MODEL_PATH = PROJECT_ROOT / "mnist_cnn.pth"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "training" / "learning_curves.png"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "training" / "training_history.json"


def _resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)


def save_final_model(model: torch.nn.Module, model_path: str | Path) -> None:
    """Save the final trained weights in the simple state-dict format."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: str | None = None) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=_resolve_device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def plot_history(history: dict[str, list[float]], plot_path: str | Path) -> None:
    """Plot loss and accuracy curves so you can include them in your report."""
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

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
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def save_history(history: dict[str, list[float]], history_path: str | Path) -> None:
    """Save training curves data so the same graph can be plotted later."""
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as file_handle:
        json.dump(history, file_handle, indent=2)


def load_history(history_path: str | Path) -> dict[str, list[float]]:
    """Load a previously saved training history file."""
    history_path = Path(history_path)
    with history_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def train_model(
    train_loader,
    val_loader,
    epochs: int = 5,
    lr: float = 1e-3,
    model_name: str = "cnn",
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    final_model_path: str | Path = DEFAULT_FINAL_MODEL_PATH,
    plot_path: str | Path = DEFAULT_PLOT_PATH,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
    device: str | None = None,
):
    """Train the CNN and save the best checkpoint by validation accuracy."""
    device = _resolve_device(device)
    model = build_model(model_name=model_name).to(device)
    criterion = get_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += accuracy_from_logits(logits, labels)
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_metrics = evaluate_model(model, val_loader, device=device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(model, checkpoint_path)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f}"
        )

    save_final_model(model, final_model_path)
    save_history(history, history_path)
    plot_history(history, plot_path)
    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Learning curves saved to: {plot_path}")
    return model, history