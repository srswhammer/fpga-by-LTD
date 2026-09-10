"""Step 7: evaluate the pure-integer network and export a first release if it passes."""

import csv
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import datasets

from integer_inference import IntegerDigitCNN
from predict import load_trained_model, preprocess_digit_image
from validate_software_prediction import make_camera_like


CANDIDATE_DIR = ROOT / "runs" / "baseline_v1" / "fixed_point_candidate_v1"
FLOAT_REF_DIR = ROOT / "runs" / "baseline_v1" / "float_reference_v1"
RELEASE_DIR = ROOT / "runs" / "digitcnn_v1_int8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def float_predict(model, images_uint8: np.ndarray, batch_size: int = 512):
    predictions = []
    scores = []
    start = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(images_uint8), batch_size):
            batch = torch.from_numpy(images_uint8[first : first + batch_size].copy())
            batch = batch.unsqueeze(1).to(torch.float32).div_(255.0)
            output = model(batch)
            scores.append(output.cpu().numpy())
            predictions.append(output.argmax(1).cpu().numpy())
    return (
        np.concatenate(predictions).astype(np.int64),
        np.concatenate(scores).astype(np.float32),
        time.perf_counter() - start,
    )


def integer_predict(model, images_uint8: np.ndarray, batch_size: int = 512):
    predictions, scores = [], []
    saturation_count = 0
    activation_count = 0
    rounded_max = 0
    conv_min, conv_max = np.iinfo(np.int32).max, np.iinfo(np.int32).min
    score_min, score_max = np.iinfo(np.int32).max, np.iinfo(np.int32).min
    start = time.perf_counter()
    for first in range(0, len(images_uint8), batch_size):
        result = model.forward(images_uint8[first : first + batch_size], return_layers=True)
        predictions.append(result.predictions)
        scores.append(result.scores_int32)
        rounded = result.relu_rounded_before_saturation
        saturation_count += int(np.count_nonzero(rounded > 255))
        activation_count += int(rounded.size)
        rounded_max = max(rounded_max, int(rounded.max()))
        conv_min = min(conv_min, int(result.conv_acc_int32.min()))
        conv_max = max(conv_max, int(result.conv_acc_int32.max()))
        score_min = min(score_min, int(result.scores_int32.min()))
        score_max = max(score_max, int(result.scores_int32.max()))
    return (
        np.concatenate(predictions),
        np.concatenate(scores),
        {
            "reference_cpu_seconds": time.perf_counter() - start,
            "relu_saturation_count": saturation_count,
            "relu_activation_count": activation_count,
            "relu_saturation_rate": saturation_count / activation_count,
            "relu_rounded_max_before_saturation": rounded_max,
            "conv_acc_actual_min": conv_min,
            "conv_acc_actual_max": conv_max,
            "score_actual_min": score_min,
            "score_actual_max": score_max,
        },
    )


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)


def evaluate_split(name, images, labels, float_model, integer_model):
    float_predictions, float_scores, float_seconds = float_predict(float_model, images)
    int_predictions, int_scores, integer_stats = integer_predict(integer_model, images)
    float_correct = int(np.count_nonzero(float_predictions == labels))
    int_correct = int(np.count_nonzero(int_predictions == labels))
    agreement = int(np.count_nonzero(float_predictions == int_predictions))
    count = len(labels)
    metrics = {
        "name": name,
        "count": count,
        "float_correct": float_correct,
        "float_accuracy": float_correct / count,
        "int8_correct": int_correct,
        "int8_accuracy": int_correct / count,
        "int8_minus_float_accuracy_points": (int_correct - float_correct) * 100.0 / count,
        "top1_agreement_count": agreement,
        "top1_agreement_rate": agreement / count,
        "float_reference_cpu_seconds": float_seconds,
        **integer_stats,
        "timing_warning": "NumPy/PyTorch PC reference timing is not FPGA latency or throughput.",
    }
    return metrics, float_predictions, float_scores, int_predictions, int_scores


def write_mem(path: Path, values: np.ndarray, bits: int):
    mask = (1 << bits) - 1
    width = bits // 4
    flat = values.reshape(-1).astype(np.int64)
    path.write_text("".join(f"{int(value) & mask:0{width}x}\n" for value in flat), encoding="ascii")


def save_npy(path: Path, value: np.ndarray, records: dict):
    value = np.ascontiguousarray(value)
    np.save(path, value, allow_pickle=False)
    records[str(path.relative_to(RELEASE_DIR)).replace("\\", "/")] = {
        "sha256": sha256_file(path),
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "array_order": "C / row-major",
    }


def main():
    candidate_config = json.loads(
        (CANDIDATE_DIR / "candidate_config.json").read_text(encoding="utf-8")
    )
    integer_model = IntegerDigitCNN.from_directory(CANDIDATE_DIR)
    float_model = load_trained_model()
    checkpoint = torch.load(
        ROOT / "runs" / "baseline_v1" / "best_model.pt", map_location="cpu", weights_only=True
    )
    torch.set_num_threads(int(checkpoint["config"]["threads"]))

    train_pool = datasets.MNIST(ROOT / "data", train=True, download=False)
    split = np.load(ROOT / "runs" / "baseline_v1" / "split_indices.npz")
    validation_indices = split["validation"]
    validation_images = train_pool.data[torch.from_numpy(validation_indices)].numpy()
    validation_labels = train_pool.targets[torch.from_numpy(validation_indices)].numpy().astype(np.int64)

    test_pool = datasets.MNIST(ROOT / "data", train=False, download=False)
    test_images = test_pool.data.numpy()
    test_labels = test_pool.targets.numpy().astype(np.int64)

    synthetic_images = []
    synthetic_labels = []
    for sample_index, dataset_index in enumerate(validation_indices[:1000]):
        source, label = train_pool[int(dataset_index)]
        camera_like: Image.Image = make_camera_like(source, sample_index)
        synthetic_images.append(preprocess_digit_image(camera_like).normalized_uint8)
        synthetic_labels.append(int(label))
    synthetic_images = np.stack(synthetic_images).astype(np.uint8)
    synthetic_labels = np.asarray(synthetic_labels, dtype=np.int64)

    results = {}
    prediction_data = {}
    for name, images, labels in (
        ("validation", validation_images, validation_labels),
        ("official_test", test_images, test_labels),
        ("synthetic_camera_like", synthetic_images, synthetic_labels),
    ):
        metrics, float_pred, float_scores, int_pred, int_scores = evaluate_split(
            name, images, labels, float_model, integer_model
        )
        results[name] = metrics
        prediction_data[name] = {
            "labels": labels,
            "float_predictions": float_pred,
            "float_scores": float_scores,
            "int8_predictions": int_pred,
            "int8_scores": int_scores,
        }
        print(
            f"{name}: float={metrics['float_accuracy']:.2%}, "
            f"int8={metrics['int8_accuracy']:.2%}, "
            f"agreement={metrics['top1_agreement_rate']:.2%}, "
            f"saturation={metrics['relu_saturation_count']}"
        )

    fixed_inputs = np.load(
        FLOAT_REF_DIR / "inputs" / "fixed_inputs_uint8.npy", allow_pickle=False
    )
    fixed_labels = np.load(FLOAT_REF_DIR / "inputs" / "labels_int64.npy", allow_pickle=False)
    fixed = integer_model.forward(fixed_inputs, return_layers=True)
    fixed_all_correct = bool(np.array_equal(fixed.predictions, fixed_labels))

    thresholds = {
        "validation_max_accuracy_drop_points": 0.5,
        "official_test_max_accuracy_drop_points": 0.5,
        "synthetic_camera_like_max_accuracy_drop_points": 1.0,
        "fixed_0_to_9_all_correct_required": True,
    }
    checks = {
        "validation_accuracy": results["validation"]["int8_minus_float_accuracy_points"] >= -0.5,
        "official_test_accuracy": results["official_test"]["int8_minus_float_accuracy_points"] >= -0.5,
        "synthetic_camera_like_accuracy": results["synthetic_camera_like"][
            "int8_minus_float_accuracy_points"
        ]
        >= -1.0,
        "fixed_0_to_9_all_correct": fixed_all_correct,
    }
    accepted = all(checks.values())
    candidate_result = {
        "candidate": candidate_config["format_version"],
        "acceptance_thresholds": thresholds,
        "acceptance_checks": checks,
        "accepted_as_digitcnn_v1_int8": accepted,
        "results": results,
    }
    (CANDIDATE_DIR / "accuracy_evaluation.json").write_text(
        json.dumps(candidate_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not accepted:
        raise RuntimeError("Fixed-point candidate failed acceptance criteria; release was not created")

    parameter_dir = RELEASE_DIR / "parameters"
    reference_dir = RELEASE_DIR / "reference"
    evaluation_dir = RELEASE_DIR / "evaluation"
    for directory in (RELEASE_DIR, parameter_dir, reference_dir, evaluation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    records = {}

    candidate_params = np.load(
        CANDIDATE_DIR / "candidate_parameters_unvalidated.npz", allow_pickle=False
    )
    parameters = {name: np.ascontiguousarray(candidate_params[name]) for name in candidate_params.files}
    parameter_archive = parameter_dir / "model_parameters_int.npz"
    np.savez_compressed(parameter_archive, **parameters)
    records["parameters/model_parameters_int.npz"] = {
        "sha256": sha256_file(parameter_archive),
        "format": "compressed NumPy archive",
    }
    for name, value in parameters.items():
        save_npy(parameter_dir / f"{name}.npy", value, records)
        write_mem(parameter_dir / f"{name}.mem", value, 8 if value.dtype == np.int8 else 32)
        records[f"parameters/{name}.mem"] = {
            "sha256": sha256_file(parameter_dir / f"{name}.mem"),
            "format": "one two's-complement hexadecimal integer per line",
        }

    binary_parts = [
        parameters["conv_weight_int8"].astype(np.int8).tobytes(order="C"),
        parameters["conv_bias_int32"].astype("<i4").tobytes(order="C"),
        parameters["fc_weight_int8"].astype(np.int8).tobytes(order="C"),
        parameters["fc_bias_int32"].astype("<i4").tobytes(order="C"),
    ]
    packed_path = parameter_dir / "model_parameters_v1.bin"
    packed_path.write_bytes(b"".join(binary_parts))
    layout = {
        "endianness": "little for int32; int8 is one byte",
        "total_bytes": len(packed_path.read_bytes()),
        "segments": [
            {"name": "conv_weight_int8", "offset": 0, "bytes": 36, "shape": [4, 1, 3, 3]},
            {"name": "conv_bias_int32", "offset": 36, "bytes": 16, "shape": [4]},
            {"name": "fc_weight_int8", "offset": 52, "bytes": 6760, "shape": [10, 676]},
            {"name": "fc_bias_int32", "offset": 6812, "bytes": 40, "shape": [10]},
        ],
    }
    (parameter_dir / "binary_layout.json").write_text(
        json.dumps(layout, indent=2), encoding="utf-8"
    )
    for relative in ("parameters/model_parameters_v1.bin", "parameters/binary_layout.json"):
        records[relative] = {"sha256": sha256_file(RELEASE_DIR / relative), "format": "binary" if relative.endswith(".bin") else "JSON"}

    save_npy(reference_dir / "fixed_inputs_uint8.npy", fixed_inputs, records)
    save_npy(reference_dir / "labels_int64.npy", fixed_labels, records)
    save_npy(reference_dir / "conv_acc_int32.npy", fixed.conv_acc_int32, records)
    save_npy(reference_dir / "relu_uint8.npy", fixed.relu_uint8, records)
    save_npy(reference_dir / "pool_uint8.npy", fixed.pool_uint8, records)
    save_npy(reference_dir / "flatten_uint8.npy", fixed.flatten_uint8, records)
    save_npy(reference_dir / "scores_int32.npy", fixed.scores_int32, records)
    save_npy(reference_dir / "predictions_int64.npy", fixed.predictions, records)

    for name, data in prediction_data.items():
        path = evaluation_dir / f"{name}_predictions.npz"
        np.savez_compressed(path, **data)
        records[f"evaluation/{path.name}"] = {
            "sha256": sha256_file(path),
            "format": "labels, float predictions/scores, integer predictions/scores",
        }
        confusion = confusion_matrix(data["labels"], data["int8_predictions"])
        np.savetxt(evaluation_dir / f"{name}_confusion_int8.csv", confusion, delimiter=",", fmt="%d")
        records[f"evaluation/{name}_confusion_int8.csv"] = {
            "sha256": sha256_file(evaluation_dir / f"{name}_confusion_int8.csv"),
            "format": "10x10 CSV; rows true, columns predicted",
        }
        per_class_path = evaluation_dir / f"{name}_per_class_accuracy.csv"
        with per_class_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "digit",
                    "total",
                    "float_correct",
                    "float_accuracy",
                    "int8_correct",
                    "int8_accuracy",
                    "int8_minus_float_accuracy_points",
                ],
            )
            writer.writeheader()
            for digit in range(10):
                mask = data["labels"] == digit
                total = int(mask.sum())
                float_correct = int(np.count_nonzero(data["float_predictions"][mask] == digit))
                int_correct = int(np.count_nonzero(data["int8_predictions"][mask] == digit))
                writer.writerow(
                    {
                        "digit": digit,
                        "total": total,
                        "float_correct": float_correct,
                        "float_accuracy": float_correct / total,
                        "int8_correct": int_correct,
                        "int8_accuracy": int_correct / total,
                        "int8_minus_float_accuracy_points": (int_correct - float_correct)
                        * 100.0
                        / total,
                    }
                )
        records[f"evaluation/{per_class_path.name}"] = {
            "sha256": sha256_file(per_class_path),
            "format": "UTF-8 CSV per-class float32 and int8 accuracy",
        }

    summary_path = evaluation_dir / "summary.json"
    summary_path.write_text(json.dumps(candidate_result, indent=2, ensure_ascii=False), encoding="utf-8")
    records["evaluation/summary.json"] = {"sha256": sha256_file(summary_path), "format": "JSON"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["Validation", "Official test", "Synthetic camera-like"]
    keys = ["validation", "official_test", "synthetic_camera_like"]
    x = np.arange(3)
    width = 0.36
    ax.bar(x - width / 2, [results[k]["float_accuracy"] * 100 for k in keys], width, label="float32")
    ax.bar(x + width / 2, [results[k]["int8_accuracy"] * 100 for k in keys], width, label="int8 integer")
    ax.set_xticks(x, names)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(88, 97)
    ax.set_title("digitcnn_v1: float32 and pure-integer accuracy")
    ax.legend()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(evaluation_dir / "accuracy_comparison.png", dpi=160)
    plt.close(fig)
    records["evaluation/accuracy_comparison.png"] = {
        "sha256": sha256_file(evaluation_dir / "accuracy_comparison.png"),
        "format": "PNG visualization",
    }

    test_data = prediction_data["official_test"]
    test_confusion = confusion_matrix(test_data["labels"], test_data["int8_predictions"])
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(test_confusion, cmap="Blues")
    ax.set(
        xlabel="Predicted digit",
        ylabel="True digit",
        xticks=range(10),
        yticks=range(10),
        title=f"digitcnn_v1_int8 test confusion - {results['official_test']['int8_accuracy']:.2%}",
    )
    threshold = test_confusion.max() / 2
    for row in range(10):
        for column in range(10):
            value = int(test_confusion[row, column])
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
    confusion_png = evaluation_dir / "official_test_confusion_int8.png"
    fig.savefig(confusion_png, dpi=160)
    plt.close(fig)
    records["evaluation/official_test_confusion_int8.png"] = {
        "sha256": sha256_file(confusion_png),
        "format": "PNG visualization",
    }

    disagreement_indices = np.flatnonzero(
        test_data["float_predictions"] != test_data["int8_predictions"]
    )
    fig, axes = plt.subplots(4, 6, figsize=(10, 7))
    for ax, index in zip(axes.flat, disagreement_indices[:24]):
        ax.imshow(test_images[index], cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"true {test_labels[index]} / f {test_data['float_predictions'][index]} "
            f"/ q {test_data['int8_predictions'][index]}",
            fontsize=8,
        )
        ax.axis("off")
    for ax in axes.flat[len(disagreement_indices[:24]) :]:
        ax.axis("off")
    fig.suptitle(f"First quantization disagreements ({len(disagreement_indices)} of 10000)")
    fig.tight_layout()
    disagreement_png = evaluation_dir / "official_test_quantization_disagreements.png"
    fig.savefig(disagreement_png, dpi=160)
    plt.close(fig)
    records["evaluation/official_test_quantization_disagreements.png"] = {
        "sha256": sha256_file(disagreement_png),
        "format": "PNG visualization",
    }

    release_config = {
        **candidate_config,
        "format_version": "digitcnn_v1_int8",
        "status": "FIRST_INTEGER_SOFTWARE_RELEASE_READY_FOR_TEAM_REVIEW",
        "parameter_file": "parameters/model_parameters_int.npz",
        "source_candidate": candidate_config["format_version"],
        "acceptance": candidate_result,
        "important_boundary": "Software integer reference is verified; FPGA synthesis, timing, resources, and board results are not yet available.",
    }
    (RELEASE_DIR / "config.json").write_text(
        json.dumps(release_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    records["config.json"] = {"sha256": sha256_file(RELEASE_DIR / "config.json"), "format": "JSON"}
    manifest = {
        "release": "digitcnn_v1_int8",
        "file_count": len(records),
        "files": records,
    }
    (RELEASE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("ACCEPTED: digitcnn_v1_int8")
    print(RELEASE_DIR)


if __name__ == "__main__":
    main()
