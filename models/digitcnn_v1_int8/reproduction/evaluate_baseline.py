"""Step 3: evaluate the selected float32 checkpoint on MNIST test data."""

import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitCNN


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs" / "baseline_v1"
EVAL_DIR = RUN_DIR / "evaluation"


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Refuse to evaluate a checkpoint if model.py changed after training.
    checkpoint = torch.load(RUN_DIR / "best_model.pt", map_location="cpu", weights_only=True)
    current_hash = hashlib.sha256((ROOT / "model.py").read_bytes()).hexdigest()
    assert checkpoint["config"]["model_sha256"] == current_hash, (
        "model.py differs from the file used to train this checkpoint"
    )

    torch.set_num_threads(checkpoint["config"]["threads"])
    model = DigitCNN().eval()
    model.load_state_dict(checkpoint["model_state_dict"])

    # This is the first stage that loads the official MNIST test split.
    test_dataset = datasets.MNIST(
        ROOT / "data", train=False, download=False, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    confusion = torch.zeros((10, 10), dtype=torch.int64)
    all_scores, all_labels, all_predictions = [], [], []
    wrong_images, wrong_truth, wrong_prediction = [], [], []
    start = time.perf_counter()

    with torch.inference_mode():
        for images, labels in test_loader:
            scores = model(images)
            predictions = scores.argmax(dim=1)
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            encoded = labels * 10 + predictions
            confusion += torch.bincount(encoded, minlength=100).reshape(10, 10)
            if len(wrong_images) < 24:
                for image, truth, prediction in zip(images, labels, predictions):
                    if truth != prediction:
                        wrong_images.append(image.cpu())
                        wrong_truth.append(int(truth))
                        wrong_prediction.append(int(prediction))
                        if len(wrong_images) == 24:
                            break

    elapsed = time.perf_counter() - start
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    predictions = torch.cat(all_predictions)
    assert tuple(scores.shape) == (10000, 10)
    assert confusion.sum().item() == 10000
    assert torch.equal(confusion.sum(dim=1), torch.bincount(labels, minlength=10))

    correct = int((predictions == labels).sum())
    accuracy = correct / len(labels)
    per_class_total = confusion.sum(dim=1)
    per_class_correct = confusion.diag()
    per_class_accuracy = per_class_correct.float() / per_class_total

    np.savetxt(EVAL_DIR / "confusion_matrix.csv", confusion.numpy(), delimiter=",", fmt="%d")
    np.savez_compressed(
        EVAL_DIR / "test_predictions.npz",
        labels=labels.numpy(),
        predictions=predictions.numpy(),
        scores=scores.numpy(),
    )
    with (EVAL_DIR / "per_class_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["digit", "correct", "total", "accuracy"]
        )
        writer.writeheader()
        for digit in range(10):
            writer.writerow(
                {
                    "digit": digit,
                    "correct": int(per_class_correct[digit]),
                    "total": int(per_class_total[digit]),
                    "accuracy": float(per_class_accuracy[digit]),
                }
            )

    summary = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_accuracy": float(checkpoint["val_accuracy"]),
        "test_correct": correct,
        "test_total": len(labels),
        "test_accuracy": accuracy,
        "test_errors": len(labels) - correct,
        "evaluation_seconds_cpu": elapsed,
        "official_test_set_used_for_model_selection": False,
        "confusion_rows": "true digit",
        "confusion_columns": "predicted digit",
    }
    (EVAL_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(confusion.numpy(), cmap="Blues")
    ax.set(
        xlabel="Predicted digit",
        ylabel="True digit",
        xticks=range(10),
        yticks=range(10),
        title=f"MNIST test confusion matrix - accuracy {accuracy:.2%}",
    )
    threshold = confusion.max().item() / 2
    for row in range(10):
        for column in range(10):
            value = int(confusion[row, column])
            ax.text(
                column,
                row,
                value,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 6, figsize=(10, 7))
    for ax, image, truth, prediction in zip(
        axes.flat, wrong_images, wrong_truth, wrong_prediction
    ):
        ax.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"true {truth} / pred {prediction}", fontsize=9)
        ax.axis("off")
    fig.suptitle("First 24 misclassified MNIST test images")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "misclassified_examples.png", dpi=160)
    plt.close(fig)

    print(f"Test accuracy: {accuracy:.2%} ({correct}/{len(labels)})")
    print(f"Errors: {len(labels) - correct}")
    for digit in range(10):
        print(
            f"Digit {digit}: {float(per_class_accuracy[digit]):.2%} "
            f"({int(per_class_correct[digit])}/{int(per_class_total[digit])})"
        )
    print(f"Outputs: {EVAL_DIR}")


if __name__ == "__main__":
    main()
