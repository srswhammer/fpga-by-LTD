"""Create Step 4 examples and validate the ordinary-image prediction interface."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import datasets

from predict import load_trained_model, predict_image, preprocess_digit_image


RUN_DIR = ROOT / "runs" / "baseline_v1" / "software_prediction"
EXAMPLE_DIR = ROOT / "examples" / "step4"


def make_camera_like(source: Image.Image, index: int) -> Image.Image:
    """Put a black digit on a larger white frame with deterministic scale and offset."""

    array = np.asarray(source.convert("L"), dtype=np.uint8)
    rows, columns = np.nonzero(array >= 24)
    digit = array[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    digit = 255 - digit
    target_long_side = 62 + (index % 5) * 7
    scale = target_long_side / max(digit.shape)
    new_size = (
        max(1, int(round(digit.shape[1] * scale))),
        max(1, int(round(digit.shape[0] * scale))),
    )
    digit_image = Image.fromarray(digit, mode="L").resize(new_size, Image.Resampling.BICUBIC)
    canvas = Image.new("L", (160, 120), color=255)
    x = 14 + (index * 17) % max(1, 132 - new_size[0])
    y = 10 + (index * 11) % max(1, 100 - new_size[1])
    canvas.paste(digit_image, (x, y))
    return canvas


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    model = load_trained_model()

    training_pool = datasets.MNIST(ROOT / "data", train=True, download=False)
    split = np.load(ROOT / "runs" / "baseline_v1" / "split_indices.npz")
    validation_indices = split["validation"]

    direct_correct = 0
    normalized_correct = 0
    transformed_correct = 0
    transformed_count = 1000
    batch_images, batch_labels = [], []

    # Full 5000-image validation: compare original ToTensor behavior with the new normalizer.
    for dataset_index in validation_indices:
        image, label = training_pool[int(dataset_index)]
        direct = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).float().div(255)
        batch_images.append(direct.unsqueeze(0))
        batch_labels.append(int(label))
    direct_batch = torch.stack(batch_images)
    labels = torch.tensor(batch_labels)
    with torch.inference_mode():
        direct_predictions = model(direct_batch).argmax(1)
    direct_correct = int((direct_predictions == labels).sum())

    normalized_tensors = []
    for dataset_index in validation_indices:
        image, _ = training_pool[int(dataset_index)]
        normalized_tensors.append(preprocess_digit_image(image).tensor[0])
    normalized_batch = torch.stack(normalized_tensors)
    with torch.inference_mode():
        normalized_predictions = model(normalized_batch).argmax(1)
    normalized_correct = int((normalized_predictions == labels).sum())

    transformed_predictions = []
    transformed_labels = []
    sample_rows = []
    for sample_index, dataset_index in enumerate(validation_indices[:transformed_count]):
        image, label = training_pool[int(dataset_index)]
        camera_like = make_camera_like(image, sample_index)
        result = predict_image(camera_like, model=model)
        transformed_predictions.append(result.digit)
        transformed_labels.append(int(label))
        if sample_index < 12:
            path = EXAMPLE_DIR / f"camera_like_{sample_index:02d}_label_{label}.png"
            camera_like.save(path)
            sample_rows.append((path, camera_like.copy(), result, int(label)))
    transformed_correct = int(
        (np.asarray(transformed_predictions) == np.asarray(transformed_labels)).sum()
    )

    # Input-type and failure-path checks.
    first_path, first_image, first_result, _ = sample_rows[0]
    from_path = predict_image(first_path, model=model)
    from_pil = predict_image(first_image, model=model)
    from_numpy = predict_image(np.asarray(first_image), model=model)
    assert from_path.digit == from_pil.digit == from_numpy.digit
    opposite_polarity = preprocess_digit_image(ImageOps.invert(first_image))
    assert np.array_equal(
        from_path.preprocessing.normalized_uint8, opposite_polarity.normalized_uint8
    )
    assert from_path.preprocessing.normalized_uint8.shape == (28, 28)
    assert from_path.preprocessing.normalized_uint8.dtype == np.uint8
    assert tuple(from_path.preprocessing.tensor.shape) == (1, 1, 28, 28)
    assert from_path.preprocessing.tensor.dtype == torch.float32
    assert 0.0 <= float(from_path.preprocessing.tensor.min()) <= 1.0
    assert 0.0 <= float(from_path.preprocessing.tensor.max()) <= 1.0
    assert np.isclose(from_path.probabilities.sum(), 1.0, atol=1e-6)
    error_checks = {}
    try:
        preprocess_digit_image(np.full((100, 100), 255, dtype=np.uint8))
    except ValueError as error:
        blank_rejected = True
        blank_error = str(error)
    else:
        raise AssertionError("Blank image should have been rejected")
    for name, bad_source, expected_error in [
        ("bad_shape", np.zeros((10, 10, 2), dtype=np.uint8), ValueError),
        ("nan_pixel", np.full((10, 10), np.nan), ValueError),
        ("missing_file", EXAMPLE_DIR / "does_not_exist.png", FileNotFoundError),
    ]:
        try:
            preprocess_digit_image(bad_source)
        except expected_error:
            error_checks[name] = True
        else:
            raise AssertionError(f"{name} input should have raised {expected_error.__name__}")

    np.save(RUN_DIR / "sample_normalized_uint8.npy", from_path.preprocessing.normalized_uint8)
    (RUN_DIR / "sample_prediction.json").write_text(
        json.dumps(
            {
                "source_file": first_path.name,
                "predicted_digit": from_path.digit,
                "scores": from_path.scores.tolist(),
                "probabilities": from_path.probabilities.tolist(),
                "preprocessing_metadata": from_path.preprocessing.metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(4, 6, figsize=(12, 8))
    for column, (_, camera_like, result, label) in enumerate(sample_rows[:6]):
        axes[0, column].imshow(camera_like, cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(f"source / true {label}")
        axes[1, column].imshow(result.preprocessing.contrast_image, cmap="gray", vmin=0, vmax=255)
        axes[1, column].set_title("polarity fixed")
        axes[2, column].imshow(result.preprocessing.cropped_image, cmap="gray", vmin=0, vmax=255)
        axes[2, column].set_title("cropped")
        axes[3, column].imshow(result.preprocessing.normalized_uint8, cmap="gray", vmin=0, vmax=255)
        confidence = float(result.probabilities[result.digit])
        axes[3, column].set_title(f"28x28: {result.digit} ({confidence:.1%})")
        for row in range(4):
            axes[row, column].axis("off")
    fig.suptitle("Step 4 preprocessing: ordinary image to model input")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "preprocessing_stages.png", dpi=160)
    plt.close(fig)

    summary = {
        "validation_total": int(len(validation_indices)),
        "direct_totensor_correct": direct_correct,
        "direct_totensor_accuracy": direct_correct / len(validation_indices),
        "normalized_correct": normalized_correct,
        "normalized_accuracy": normalized_correct / len(validation_indices),
        "camera_like_total": transformed_count,
        "camera_like_correct": transformed_correct,
        "camera_like_accuracy": transformed_correct / transformed_count,
        "camera_like_transform_is_synthetic": True,
        "input_types_checked": ["path", "PIL.Image", "NumPy array"],
        "automatic_black_white_polarity_equivalence": True,
        "output_uint8_shape": [28, 28],
        "output_tensor_shape": [1, 1, 28, 28],
        "blank_image_rejected": blank_rejected,
        "blank_image_error": blank_error,
        "other_error_checks": error_checks,
        "sample_uint8_saved": "sample_normalized_uint8.npy",
        "checkpoint": str((ROOT / "runs" / "baseline_v1" / "best_model.pt").resolve()),
        "note": "Validation split only; preprocessing has not been validated with real OV5640 captures.",
    }
    (RUN_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
