"""Verify hashes, parameter encodings, fixed layer outputs, and evaluation reports."""

import hashlib
import json
from pathlib import Path

import numpy as np

from integer_inference import IntegerDigitCNN


ROOT = Path(__file__).resolve().parent
RELEASE_DIR = ROOT / "runs" / "digitcnn_v1_int8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mem(path: Path, dtype) -> np.ndarray:
    values = np.asarray([int(line, 16) for line in path.read_text(encoding="ascii").splitlines()])
    if dtype == np.int8:
        return values.astype(np.uint8).view(np.int8)
    return values.astype(np.uint32).view(np.int32)


def main():
    manifest = json.loads((RELEASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((RELEASE_DIR / "config.json").read_text(encoding="utf-8"))
    hashes_ok = all(
        (RELEASE_DIR / relative).is_file()
        and sha256_file(RELEASE_DIR / relative) == record["sha256"]
        for relative, record in manifest["files"].items()
    )
    if not hashes_ok:
        raise AssertionError("Release manifest hash check failed")

    archive = np.load(RELEASE_DIR / config["parameter_file"], allow_pickle=False)
    parameters = {name: archive[name] for name in archive.files}
    npy_copies_exact = all(
        np.array_equal(
            value,
            np.load(RELEASE_DIR / "parameters" / f"{name}.npy", allow_pickle=False),
        )
        for name, value in parameters.items()
    )
    mem_copies_exact = all(
        np.array_equal(
            value.reshape(-1),
            read_mem(
                RELEASE_DIR / "parameters" / f"{name}.mem",
                np.int8 if value.dtype == np.int8 else np.int32,
            ),
        )
        for name, value in parameters.items()
    )

    binary = (RELEASE_DIR / "parameters" / "model_parameters_v1.bin").read_bytes()
    layout = json.loads(
        (RELEASE_DIR / "parameters" / "binary_layout.json").read_text(encoding="utf-8")
    )
    binary_arrays = {
        "conv_weight_int8": np.frombuffer(binary, dtype=np.int8, count=36, offset=0).reshape(4, 1, 3, 3),
        "conv_bias_int32": np.frombuffer(binary, dtype="<i4", count=4, offset=36),
        "fc_weight_int8": np.frombuffer(binary, dtype=np.int8, count=6760, offset=52).reshape(10, 676),
        "fc_bias_int32": np.frombuffer(binary, dtype="<i4", count=10, offset=6812),
    }
    binary_copies_exact = len(binary) == layout["total_bytes"] == 6852 and all(
        np.array_equal(parameters[name], value) for name, value in binary_arrays.items()
    )

    model = IntegerDigitCNN.from_directory(RELEASE_DIR)
    reference_dir = RELEASE_DIR / "reference"
    inputs = np.load(reference_dir / "fixed_inputs_uint8.npy", allow_pickle=False)
    labels = np.load(reference_dir / "labels_int64.npy", allow_pickle=False)
    result = model.forward(inputs, return_layers=True)
    layer_outputs_exact = {
        "conv_acc": np.array_equal(
            result.conv_acc_int32, np.load(reference_dir / "conv_acc_int32.npy", allow_pickle=False)
        ),
        "relu": np.array_equal(
            result.relu_uint8, np.load(reference_dir / "relu_uint8.npy", allow_pickle=False)
        ),
        "pool": np.array_equal(
            result.pool_uint8, np.load(reference_dir / "pool_uint8.npy", allow_pickle=False)
        ),
        "flatten": np.array_equal(
            result.flatten_uint8, np.load(reference_dir / "flatten_uint8.npy", allow_pickle=False)
        ),
        "scores": np.array_equal(
            result.scores_int32, np.load(reference_dir / "scores_int32.npy", allow_pickle=False)
        ),
        "predictions": np.array_equal(
            result.predictions, np.load(reference_dir / "predictions_int64.npy", allow_pickle=False)
        ),
    }
    fixed_samples_correct = np.array_equal(result.predictions, labels)

    summary = json.loads(
        (RELEASE_DIR / "evaluation" / "summary.json").read_text(encoding="utf-8")
    )
    evaluation_files_consistent = True
    confusion_totals = {}
    for name in ("validation", "official_test", "synthetic_camera_like"):
        data = np.load(
            RELEASE_DIR / "evaluation" / f"{name}_predictions.npz", allow_pickle=False
        )
        count = len(data["labels"])
        float_correct = int(np.count_nonzero(data["float_predictions"] == data["labels"]))
        int_correct = int(np.count_nonzero(data["int8_predictions"] == data["labels"]))
        metrics = summary["results"][name]
        evaluation_files_consistent &= count == metrics["count"]
        evaluation_files_consistent &= float_correct == metrics["float_correct"]
        evaluation_files_consistent &= int_correct == metrics["int8_correct"]
        confusion = np.loadtxt(
            RELEASE_DIR / "evaluation" / f"{name}_confusion_int8.csv",
            delimiter=",",
            dtype=np.int64,
        )
        expected_confusion = np.bincount(
            data["labels"] * 10 + data["int8_predictions"], minlength=100
        ).reshape(10, 10)
        evaluation_files_consistent &= np.array_equal(confusion, expected_confusion)
        confusion_totals[name] = int(confusion.sum())

    report = {
        "verification_passed": all(
            [
                hashes_ok,
                npy_copies_exact,
                mem_copies_exact,
                binary_copies_exact,
                *layer_outputs_exact.values(),
                fixed_samples_correct,
                evaluation_files_consistent,
                summary["accepted_as_digitcnn_v1_int8"],
            ]
        ),
        "release_status": config["status"],
        "manifest_hashes_ok": hashes_ok,
        "npy_parameter_copies_exact": npy_copies_exact,
        "verilog_mem_parameter_copies_exact": mem_copies_exact,
        "packed_binary_parameter_copies_exact": binary_copies_exact,
        "packed_binary_bytes": len(binary),
        "fixed_layer_outputs_recomputed_exactly": layer_outputs_exact,
        "fixed_0_to_9_all_correct": fixed_samples_correct,
        "evaluation_files_consistent": evaluation_files_consistent,
        "confusion_matrix_totals": confusion_totals,
        "accuracy_acceptance_passed": summary["accepted_as_digitcnn_v1_int8"],
        "fpga_hardware_verified": False,
    }
    if not report["verification_passed"]:
        raise AssertionError(json.dumps(report, indent=2))
    (RELEASE_DIR / "verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
