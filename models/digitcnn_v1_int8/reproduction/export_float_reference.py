"""Step 5: export deterministic float32 weights, inputs, and layer references."""

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets

from predict import DEFAULT_CHECKPOINT, load_trained_model, preprocess_digit_image


EXPORT_DIR = ROOT / "runs" / "baseline_v1" / "float_reference_v1"
INPUT_DIR = EXPORT_DIR / "inputs"
WEIGHT_DIR = EXPORT_DIR / "weights"
ACTIVATION_DIR = EXPORT_DIR / "activations"
MANUAL_DIR = EXPORT_DIR / "manual_check"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    for directory in (EXPORT_DIR, INPUT_DIR, WEIGHT_DIR, ACTIVATION_DIR, MANUAL_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    torch.set_num_threads(int(checkpoint["config"]["threads"]))
    torch.use_deterministic_algorithms(True)
    model = load_trained_model()

    # Select the first correctly classified validation example for each digit.
    pool = datasets.MNIST(ROOT / "data", train=True, download=False)
    split = np.load(ROOT / "runs" / "baseline_v1" / "split_indices.npz")
    validation_indices = split["validation"]
    selected = {}
    for validation_position, dataset_index in enumerate(validation_indices):
        image, label = pool[int(dataset_index)]
        if int(label) in selected:
            continue
        prepared = preprocess_digit_image(image)
        with torch.inference_mode():
            prediction = int(model(prepared.tensor).argmax(1).item())
        if prediction == int(label):
            selected[int(label)] = {
                "validation_position": validation_position,
                "dataset_index": int(dataset_index),
                "input_uint8": prepared.normalized_uint8,
            }
        if len(selected) == 10:
            break
    if sorted(selected) != list(range(10)):
        raise RuntimeError("Could not select one correct validation example for every digit")

    labels = np.arange(10, dtype=np.int64)
    source_indices = np.asarray([selected[d]["dataset_index"] for d in range(10)], dtype=np.int64)
    validation_positions = np.asarray(
        [selected[d]["validation_position"] for d in range(10)], dtype=np.int64
    )
    inputs_uint8 = np.stack([selected[d]["input_uint8"] for d in range(10)]).astype(np.uint8)
    inputs_float32 = inputs_uint8.astype(np.float32) / np.float32(255.0)
    inputs_float32 = inputs_float32[:, np.newaxis, :, :]
    input_tensor = torch.from_numpy(inputs_float32.copy())

    # Run each layer explicitly so every boundary has a named reference value.
    with torch.inference_mode():
        conv = model.conv(input_tensor)
        relu = model.relu(conv)
        pool_output = model.pool(relu)
        flatten = model.flatten(pool_output)
        scores = model.fc(flatten)
    predictions = scores.argmax(dim=1)
    assert torch.equal(predictions, torch.from_numpy(labels))

    arrays = {
        "inputs/fixed_inputs_uint8.npy": inputs_uint8,
        "inputs/fixed_inputs_float32.npy": inputs_float32,
        "inputs/labels_int64.npy": labels,
        "inputs/source_indices_int64.npy": source_indices,
        "inputs/validation_positions_int64.npy": validation_positions,
        "weights/conv_weight_float32.npy": model.conv.weight.detach().cpu().numpy(),
        "weights/conv_bias_float32.npy": model.conv.bias.detach().cpu().numpy(),
        "weights/fc_weight_float32.npy": model.fc.weight.detach().cpu().numpy(),
        "weights/fc_bias_float32.npy": model.fc.bias.detach().cpu().numpy(),
        "activations/conv_output_float32.npy": conv.cpu().numpy(),
        "activations/relu_output_float32.npy": relu.cpu().numpy(),
        "activations/pool_output_float32.npy": pool_output.cpu().numpy(),
        "activations/flatten_output_float32.npy": flatten.cpu().numpy(),
        "activations/scores_float32.npy": scores.cpu().numpy(),
        "activations/predictions_int64.npy": predictions.cpu().numpy(),
    }
    file_records = {}
    for relative_name, array in arrays.items():
        path = EXPORT_DIR / relative_name
        contiguous = np.ascontiguousarray(array)
        np.save(path, contiguous, allow_pickle=False)
        file_records[relative_name] = {
            "sha256": sha256_file(path),
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "array_order": "C / row-major",
        }

    archive_path = EXPORT_DIR / "reference_bundle_v1.npz"
    np.savez_compressed(archive_path, **{name.replace("/", "__").removesuffix(".npy"): value for name, value in arrays.items()})
    file_records[archive_path.name] = {
        "sha256": sha256_file(archive_path),
        "format": "compressed NumPy archive containing copies of all named arrays",
    }

    with (EXPORT_DIR / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sample_id", "true_label", "prediction", "mnist_train_index", "validation_position"],
        )
        writer.writeheader()
        for sample_id in range(10):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "true_label": int(labels[sample_id]),
                    "prediction": int(predictions[sample_id]),
                    "mnist_train_index": int(source_indices[sample_id]),
                    "validation_position": int(validation_positions[sample_id]),
                }
            )
    file_records["samples.csv"] = {
        "sha256": sha256_file(EXPORT_DIR / "samples.csv"),
        "format": "UTF-8 CSV sample index and labels",
    }

    # Pick a non-empty 3x3 patch from digit 0 and record all nine multiply terms.
    sample_id = 0
    input_image = inputs_float32[sample_id, 0]
    patch_sums = np.asarray(
        [
            [input_image[row : row + 3, column : column + 3].sum() for column in range(26)]
            for row in range(26)
        ]
    )
    output_row, output_column = np.unravel_index(int(patch_sums.argmax()), patch_sums.shape)
    output_channel = int(
        np.abs(conv[sample_id, :, output_row, output_column].cpu().numpy()).argmax()
    )
    patch = inputs_float32[sample_id, 0, output_row : output_row + 3, output_column : output_column + 3]
    kernel = arrays["weights/conv_weight_float32.npy"][output_channel, 0]
    bias = np.float32(arrays["weights/conv_bias_float32.npy"][output_channel])
    accumulator = np.float32(bias)
    terms = []
    for kernel_row in range(3):
        for kernel_column in range(3):
            input_value = np.float32(patch[kernel_row, kernel_column])
            weight_value = np.float32(kernel[kernel_row, kernel_column])
            product = np.float32(input_value * weight_value)
            accumulator = np.float32(accumulator + product)
            terms.append(
                {
                    "kernel_row": kernel_row,
                    "kernel_column": kernel_column,
                    "input": float(input_value),
                    "weight": float(weight_value),
                    "product_float32": float(product),
                    "accumulator_after_float32": float(accumulator),
                }
            )
    torch_value = float(conv[sample_id, output_channel, output_row, output_column])
    manual = {
        "sample_id": sample_id,
        "true_digit": int(labels[sample_id]),
        "output_channel": output_channel,
        "output_row": int(output_row),
        "output_column": int(output_column),
        "input_patch_rows": [int(output_row), int(output_row + 3)],
        "input_patch_columns": [int(output_column), int(output_column + 3)],
        "bias": float(bias),
        "manual_float32_sequential_result": float(accumulator),
        "pytorch_conv_result": torch_value,
        "absolute_difference": abs(float(accumulator) - torch_value),
        "equation": "bias + sum(input[kernel_row,kernel_column] * weight[kernel_row,kernel_column])",
    }
    if manual["absolute_difference"] > 1e-6:
        raise AssertionError("Manual convolution differs from PyTorch beyond float32 rounding tolerance")
    (MANUAL_DIR / "manual_conv.json").write_text(
        json.dumps(manual, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (MANUAL_DIR / "manual_conv_terms.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(terms[0]))
        writer.writeheader()
        writer.writerows(terms)
    for relative_name in ("manual_check/manual_conv.json", "manual_check/manual_conv_terms.csv"):
        file_records[relative_name] = {
            "sha256": sha256_file(EXPORT_DIR / relative_name),
            "format": "UTF-8 JSON" if relative_name.endswith(".json") else "UTF-8 CSV",
        }

    # Visual overview makes accidental sample/order changes easy to notice.
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for sample_id, ax in enumerate(axes.flat):
        ax.imshow(inputs_uint8[sample_id], cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"id {sample_id}: label {labels[sample_id]} / pred {int(predictions[sample_id])}\n"
            f"MNIST index {source_indices[sample_id]}",
            fontsize=10,
        )
        ax.axis("off")
    fig.suptitle("Fixed float-reference inputs: one correct validation sample per digit")
    fig.subplots_adjust(top=0.86, bottom=0.05, hspace=0.48, wspace=0.16)
    fig.savefig(EXPORT_DIR / "fixed_inputs_overview.png", dpi=160)
    plt.close(fig)
    file_records["fixed_inputs_overview.png"] = {
        "sha256": sha256_file(EXPORT_DIR / "fixed_inputs_overview.png"),
        "format": "PNG visual overview; not consumed by inference",
    }

    manifest = {
        "format_version": "float_reference_v1",
        "model_version": "digitcnn_v1_fp32_baseline",
        "purpose": "software golden reference for later layer-by-layer hardware comparison",
        "sample_selection": "first correctly classified validation-split sample for each true digit 0..9",
        "sample_order": "sample id equals true digit: 0,1,2,3,4,5,6,7,8,9",
        "official_mnist_test_set_used": False,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_accuracy": float(checkpoint["val_accuracy"]),
        "checkpoint_sha256": sha256_file(DEFAULT_CHECKPOINT),
        "model_py_sha256": checkpoint["config"]["model_sha256"],
        "input_contract": {
            "uint8": "[N,28,28], white digit on black background, 0..255, C row-major",
            "float32": "uint8 / 255.0, [N,1,28,28] NCHW, C row-major",
        },
        "layer_order": ["conv", "relu", "pool", "flatten", "fc/scores"],
        "weight_layouts": {
            "conv_weight": "[output_channel,input_channel,kernel_row,kernel_column]",
            "conv_bias": "[output_channel]",
            "fc_weight": "[output_class,flatten_index]",
            "fc_bias": "[output_class]",
        },
        "flatten_rule": "flatten_index = channel*13*13 + row*13 + column; column changes fastest",
        "scores_rule": "ten raw scores for digits 0..9; prediction is argmax; no softmax required",
        "float_warning": "These float32 files are software references, not the final PL wire or integer format.",
        "files": file_records,
    }
    (EXPORT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Exported {len(arrays)} arrays for {len(labels)} fixed samples")
    print(f"Manual convolution absolute difference: {manual['absolute_difference']:.3e}")
    print(EXPORT_DIR)


if __name__ == "__main__":
    main()
