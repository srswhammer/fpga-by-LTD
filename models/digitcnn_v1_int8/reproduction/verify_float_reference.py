"""Independently reload and verify every Step 5 float-reference artifact."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model import DigitCNN
from predict import DEFAULT_CHECKPOINT


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "runs" / "baseline_v1" / "float_reference_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((EXPORT_DIR / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    torch.set_num_threads(int(checkpoint["config"]["threads"]))
    torch.use_deterministic_algorithms(True)

    hashes_ok = True
    array_metadata_ok = True
    for relative_name, expected in manifest["files"].items():
        path = EXPORT_DIR / relative_name
        hashes_ok &= path.is_file() and sha256_file(path) == expected["sha256"]
        if relative_name.endswith(".npy"):
            array = np.load(path, allow_pickle=False)
            array_metadata_ok &= list(array.shape) == expected["shape"]
            array_metadata_ok &= array.dtype.str == expected["dtype"]
    if not hashes_ok or not array_metadata_ok:
        raise AssertionError("Manifest hash, shape, or dtype verification failed")

    load = lambda relative: np.load(EXPORT_DIR / relative, allow_pickle=False)
    inputs_uint8 = load("inputs/fixed_inputs_uint8.npy")
    inputs_float32 = load("inputs/fixed_inputs_float32.npy")
    labels = load("inputs/labels_int64.npy")
    expected_conv = load("activations/conv_output_float32.npy")
    expected_relu = load("activations/relu_output_float32.npy")
    expected_pool = load("activations/pool_output_float32.npy")
    expected_flatten = load("activations/flatten_output_float32.npy")
    expected_scores = load("activations/scores_float32.npy")
    expected_predictions = load("activations/predictions_int64.npy")

    converted = inputs_uint8.astype(np.float32) / np.float32(255.0)
    converted = converted[:, np.newaxis, :, :]
    input_conversion_exact = np.array_equal(converted, inputs_float32)
    flatten_rule_exact = np.array_equal(expected_pool.reshape(10, -1), expected_flatten)

    exported_state = {
        "conv.weight": torch.from_numpy(load("weights/conv_weight_float32.npy").copy()),
        "conv.bias": torch.from_numpy(load("weights/conv_bias_float32.npy").copy()),
        "fc.weight": torch.from_numpy(load("weights/fc_weight_float32.npy").copy()),
        "fc.bias": torch.from_numpy(load("weights/fc_bias_float32.npy").copy()),
    }
    checkpoint_weights_exact = all(
        torch.equal(exported_state[name], checkpoint["model_state_dict"][name]) for name in exported_state
    )
    model = DigitCNN().eval()
    model.load_state_dict(exported_state)
    x = torch.from_numpy(inputs_float32.copy())
    with torch.inference_mode():
        conv = model.conv(x)
        relu = model.relu(conv)
        pool = model.pool(relu)
        flatten = model.flatten(pool)
        scores = model.fc(flatten)
    reproduced = {
        "conv": np.array_equal(conv.numpy(), expected_conv),
        "relu": np.array_equal(relu.numpy(), expected_relu),
        "pool": np.array_equal(pool.numpy(), expected_pool),
        "flatten": np.array_equal(flatten.numpy(), expected_flatten),
        "scores": np.array_equal(scores.numpy(), expected_scores),
        "predictions": np.array_equal(scores.argmax(1).numpy(), expected_predictions),
    }
    selected_samples_correct = np.array_equal(expected_predictions, labels)

    manual = json.loads(
        (EXPORT_DIR / "manual_check" / "manual_conv.json").read_text(encoding="utf-8")
    )
    sample = manual["sample_id"]
    channel = manual["output_channel"]
    row = manual["output_row"]
    column = manual["output_column"]
    patch = inputs_float32[sample, 0, row : row + 3, column : column + 3]
    kernel = exported_state["conv.weight"].numpy()[channel, 0]
    bias = exported_state["conv.bias"].numpy()[channel]
    manual_float64 = float(bias) + float(
        (patch.astype(np.float64) * kernel.astype(np.float64)).sum()
    )
    manual_close = abs(manual_float64 - float(expected_conv[sample, channel, row, column])) <= 1e-6

    archive = np.load(EXPORT_DIR / "reference_bundle_v1.npz", allow_pickle=False)
    archive_copies_exact = True
    for relative_name, expected in manifest["files"].items():
        if not relative_name.endswith(".npy"):
            continue
        key = relative_name.replace("/", "__").removesuffix(".npy")
        archive_copies_exact &= np.array_equal(archive[key], load(relative_name))

    report = {
        "verification_passed": all(
            [
                hashes_ok,
                array_metadata_ok,
                input_conversion_exact,
                flatten_rule_exact,
                checkpoint_weights_exact,
                selected_samples_correct,
                manual_close,
                archive_copies_exact,
                *reproduced.values(),
            ]
        ),
        "manifest_hashes_ok": hashes_ok,
        "array_shapes_and_dtypes_ok": array_metadata_ok,
        "uint8_to_float32_conversion_exact": input_conversion_exact,
        "flatten_rule_exact": flatten_rule_exact,
        "exported_weights_match_checkpoint_exactly": checkpoint_weights_exact,
        "reloaded_layer_outputs_bit_exact": reproduced,
        "selected_predictions_match_labels": selected_samples_correct,
        "manual_convolution_matches_within_1e-6": manual_close,
        "manual_convolution_float64_absolute_difference": abs(
            manual_float64 - float(expected_conv[sample, channel, row, column])
        ),
        "compressed_archive_copies_exact": archive_copies_exact,
    }
    if not report["verification_passed"]:
        raise AssertionError(json.dumps(report, indent=2))
    (EXPORT_DIR / "verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
