"""Verify Step 6 candidate formats and arithmetic rules without evaluating accuracy."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from predict import load_trained_model


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "runs" / "baseline_v1" / "fixed_point_candidate_v1"
FLOAT_REF = ROOT / "runs" / "baseline_v1" / "float_reference_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    config = json.loads((OUTPUT_DIR / "candidate_config.json").read_text(encoding="utf-8"))
    ranges = json.loads((OUTPUT_DIR / "range_analysis.json").read_text(encoding="utf-8"))
    example = json.loads((OUTPUT_DIR / "integer_rule_example.json").read_text(encoding="utf-8"))
    parameter_path = OUTPUT_DIR / config["parameter_file"]
    parameter_file_hash_ok = sha256_file(parameter_path) == config["parameter_file_sha256"]
    params = np.load(parameter_path, allow_pickle=False)
    model = load_trained_model()

    input_f = config["numeric_rules"]["input"]["fraction_bits"]
    conv_w_f = config["numeric_rules"]["conv_weight"]["fraction_bits"]
    act_f = config["numeric_rules"]["relu_requantized_activation"]["fraction_bits"]
    fc_w_f = config["numeric_rules"]["fc_weight"]["fraction_bits"]
    conv_acc_f = config["numeric_rules"]["conv_accumulator"]["fraction_bits"]
    fc_acc_f = config["numeric_rules"]["fc_scores"]["fraction_bits"]
    right_shift = conv_acc_f - act_f

    expected = {
        "conv_weight_int8": np.clip(
            np.rint(model.conv.weight.detach().numpy().astype(np.float64) * 2**conv_w_f),
            -127,
            127,
        ).astype(np.int8),
        "conv_bias_int32": np.rint(
            model.conv.bias.detach().numpy().astype(np.float64) * 2**conv_acc_f
        ).astype(np.int32),
        "fc_weight_int8": np.clip(
            np.rint(model.fc.weight.detach().numpy().astype(np.float64) * 2**fc_w_f),
            -127,
            127,
        ).astype(np.int8),
        "fc_bias_int32": np.rint(
            model.fc.bias.detach().numpy().astype(np.float64) * 2**fc_acc_f
        ).astype(np.int32),
    }
    parameters_exact = {name: np.array_equal(params[name], value) for name, value in expected.items()}
    shapes_and_dtypes = {
        "conv_weight": params["conv_weight_int8"].shape == (4, 1, 3, 3)
        and params["conv_weight_int8"].dtype == np.int8,
        "conv_bias": params["conv_bias_int32"].shape == (4,)
        and params["conv_bias_int32"].dtype == np.int32,
        "fc_weight": params["fc_weight_int8"].shape == (10, 676)
        and params["fc_weight_int8"].dtype == np.int8,
        "fc_bias": params["fc_bias_int32"].shape == (10,)
        and params["fc_bias_int32"].dtype == np.int32,
    }
    scale_alignment = {
        "conv_product_and_bias": conv_acc_f == input_f + conv_w_f,
        "conv_requant_shift": right_shift > 0,
        "fc_product_and_bias": fc_acc_f == act_f + fc_w_f,
        "common_fc_score_scale": config["numeric_rules"]["fc_scores"]["class_scale"].startswith(
            "one common scale"
        ),
    }

    inputs = np.load(FLOAT_REF / "inputs" / "fixed_inputs_uint8.npy", allow_pickle=False)
    sample, channel, row, column = (
        example["sample_id"],
        example["output_channel"],
        example["output_row"],
        example["output_column"],
    )
    patch = inputs[sample, row : row + 3, column : column + 3].astype(np.int64)
    kernel = params["conv_weight_int8"][channel, 0].astype(np.int64)
    accumulator = int(params["conv_bias_int32"][channel]) + int((patch * kernel).sum())
    positive = max(accumulator, 0)
    activation_q = min(255, (positive + (1 << (right_shift - 1))) >> right_shift)
    example_exact = accumulator == example["integer_accumulator"] and activation_q == example[
        "relu_uint8_result"
    ]

    report = {
        "verification_passed": all(
            [
                *parameters_exact.values(),
                *shapes_and_dtypes.values(),
                *scale_alignment.values(),
                parameter_file_hash_ok,
                example_exact,
                ranges["int32_headroom_confirmed"],
            ]
        ),
        "candidate_status_is_unvalidated": config["status"].startswith("UNVALIDATED"),
        "parameter_file_hash_ok": parameter_file_hash_ok,
        "quantized_parameters_recreated_exactly": parameters_exact,
        "parameter_shapes_and_dtypes": shapes_and_dtypes,
        "scale_alignment": scale_alignment,
        "integer_example_recreated_exactly": example_exact,
        "int32_headroom_confirmed": ranges["int32_headroom_confirmed"],
        "accuracy_evaluated_in_this_step": False,
    }
    if not report["verification_passed"]:
        raise AssertionError(json.dumps(report, indent=2))
    (OUTPUT_DIR / "verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
