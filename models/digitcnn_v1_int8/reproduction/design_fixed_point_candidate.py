"""Step 6: derive an unvalidated power-of-two fixed-point candidate from real ranges."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from predict import DEFAULT_CHECKPOINT, load_trained_model


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "runs" / "baseline_v1" / "fixed_point_candidate_v1"
FLOAT_REF = ROOT / "runs" / "baseline_v1" / "float_reference_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_signed_int8_fraction_bits(values: np.ndarray) -> int:
    """Largest nonnegative F for round(values * 2**F) to fit symmetric int8."""

    maximum = float(np.max(np.abs(values)))
    if maximum == 0:
        return 0
    fraction_bits = max(0, int(math.floor(math.log2(127.0 / maximum))))
    while np.max(np.abs(np.rint(values * (2**fraction_bits)))) > 127:
        fraction_bits -= 1
    return fraction_bits


def choose_unsigned_int8_fraction_bits(maximum: float) -> int:
    """Largest nonnegative F such that maximum * 2**F fits 0..255."""

    if maximum <= 0:
        return 0
    return max(0, int(math.floor(math.log2(255.0 / maximum))))


def signed_bits_for_absolute_bound(bound: int) -> int:
    """Smallest two's-complement width whose positive range includes bound."""

    return max(2, int(math.ceil(math.log2(bound + 1))) + 1)


def quantize_signed_int8(values: np.ndarray, fraction_bits: int) -> np.ndarray:
    quantized = np.rint(values.astype(np.float64) * (2**fraction_bits))
    return np.clip(quantized, -127, 127).astype(np.int8)


def quantize_bias_int32(values: np.ndarray, fraction_bits: int) -> np.ndarray:
    quantized = np.rint(values.astype(np.float64) * (2**fraction_bits))
    if np.any(quantized < np.iinfo(np.int32).min) or np.any(quantized > np.iinfo(np.int32).max):
        raise OverflowError("Bias does not fit int32")
    return quantized.astype(np.int32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    torch.set_num_threads(int(checkpoint["config"]["threads"]))
    model = load_trained_model()

    # Calibration reads only the 55,000 training indices; validation/test are untouched.
    pool = datasets.MNIST(
        ROOT / "data", train=True, download=False, transform=transforms.ToTensor()
    )
    split = np.load(ROOT / "runs" / "baseline_v1" / "split_indices.npz")
    calibration_indices = split["train"]
    loader = DataLoader(
        Subset(pool, calibration_indices.tolist()),
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )
    channel_max = torch.zeros(4)
    with torch.inference_mode():
        for images, _ in loader:
            activation = torch.relu(model.conv(images))
            channel_max = torch.maximum(channel_max, activation.amax(dim=(0, 2, 3)).cpu())
    activation_max = float(channel_max.max())

    conv_weight = model.conv.weight.detach().cpu().numpy()
    conv_bias = model.conv.bias.detach().cpu().numpy()
    fc_weight = model.fc.weight.detach().cpu().numpy()
    fc_bias = model.fc.bias.detach().cpu().numpy()

    input_fraction_bits = 8
    conv_weight_fraction_bits = choose_signed_int8_fraction_bits(conv_weight)
    activation_fraction_bits = choose_unsigned_int8_fraction_bits(activation_max)
    fc_weight_fraction_bits = choose_signed_int8_fraction_bits(fc_weight)
    conv_accumulator_fraction_bits = input_fraction_bits + conv_weight_fraction_bits
    fc_accumulator_fraction_bits = activation_fraction_bits + fc_weight_fraction_bits
    conv_requant_right_shift = conv_accumulator_fraction_bits - activation_fraction_bits
    if conv_requant_right_shift < 1:
        raise RuntimeError("Candidate requires a left shift; revise the runtime rule")

    conv_weight_q = quantize_signed_int8(conv_weight, conv_weight_fraction_bits)
    conv_bias_q = quantize_bias_int32(conv_bias, conv_accumulator_fraction_bits)
    fc_weight_q = quantize_signed_int8(fc_weight, fc_weight_fraction_bits)
    fc_bias_q = quantize_bias_int32(fc_bias, fc_accumulator_fraction_bits)

    conv_bounds = (
        255 * np.abs(conv_weight_q.astype(np.int64)).sum(axis=(1, 2, 3))
        + np.abs(conv_bias_q.astype(np.int64))
    )
    fc_bounds = (
        255 * np.abs(fc_weight_q.astype(np.int64)).sum(axis=1)
        + np.abs(fc_bias_q.astype(np.int64))
    )
    conv_bound = int(conv_bounds.max())
    fc_bound = int(fc_bounds.max())

    parameters_path = OUTPUT_DIR / "candidate_parameters_unvalidated.npz"
    np.savez_compressed(
        parameters_path,
        conv_weight_int8=conv_weight_q,
        conv_bias_int32=conv_bias_q,
        fc_weight_int8=fc_weight_q,
        fc_bias_int32=fc_bias_q,
    )

    config = {
        "format_version": "fixed_point_candidate_v1",
        "status": "UNVALIDATED_CANDIDATE_DO_NOT_TREAT_AS_FINAL_HARDWARE_FORMAT",
        "model_version": "digitcnn_v1_fp32_baseline",
        "calibration_source": "all 55000 training-split samples; validation and official test excluded",
        "numeric_rules": {
            "input": {
                "storage": "uint8",
                "integer_range": [0, 255],
                "fraction_bits": input_fraction_bits,
                "real_value": "input_uint8 * 2^-8",
                "note": "255 represents 0.99609375 instead of float baseline 1.0",
            },
            "conv_weight": {
                "storage": "int8 symmetric",
                "integer_range": [-127, 127],
                "fraction_bits": conv_weight_fraction_bits,
                "real_value": f"integer * 2^-{conv_weight_fraction_bits}",
                "offline_rounding": "nearest, ties to even (NumPy rint)",
            },
            "conv_bias": {
                "storage": "int32",
                "fraction_bits": conv_accumulator_fraction_bits,
                "real_value": f"integer * 2^-{conv_accumulator_fraction_bits}",
            },
            "conv_accumulator": {
                "storage": "int32",
                "fraction_bits": conv_accumulator_fraction_bits,
                "operation": "bias_int32 + sum(uint8_input * int8_weight)",
            },
            "relu_requantized_activation": {
                "storage": "uint8",
                "integer_range": [0, 255],
                "fraction_bits": activation_fraction_bits,
                "real_value": f"integer * 2^-{activation_fraction_bits}",
                "runtime_rule": (
                    f"sat_u8((max(conv_acc,0) + 2^{conv_requant_right_shift - 1}) "
                    f">> {conv_requant_right_shift})"
                ),
                "rounding": "round half upward because the ReLU input is nonnegative",
                "saturation": "clip below 0 to 0 and above 255 to 255",
            },
            "max_pool": {
                "storage": "uint8",
                "fraction_bits": activation_fraction_bits,
                "operation": "unsigned maximum over each 2x2 window; scale unchanged",
            },
            "fc_weight": {
                "storage": "int8 symmetric",
                "integer_range": [-127, 127],
                "fraction_bits": fc_weight_fraction_bits,
                "real_value": f"integer * 2^-{fc_weight_fraction_bits}",
                "offline_rounding": "nearest, ties to even (NumPy rint)",
            },
            "fc_bias": {
                "storage": "int32",
                "fraction_bits": fc_accumulator_fraction_bits,
                "real_value": f"integer * 2^-{fc_accumulator_fraction_bits}",
            },
            "fc_scores": {
                "storage": "int32",
                "fraction_bits": fc_accumulator_fraction_bits,
                "operation": "bias_int32 + sum(uint8_pool * int8_weight)",
                "class_scale": "one common scale for all ten classes, so integer argmax is valid",
            },
        },
        "layout_rules": {
            "input": "[28,28] C row-major; column changes fastest",
            "conv_weight": "[4,1,3,3] = [output_channel,input_channel,kernel_row,kernel_column]",
            "pool_output": "[4,13,13] C row-major",
            "flatten": "index = channel*169 + row*13 + column",
            "fc_weight": "[10,676] = [output_class,flatten_index]",
            "fc_scores": "[10] for digits 0..9",
        },
        "logical_transaction": {
            "input_payload": "784 uint8 bytes",
            "control": ["start", "busy", "done", "optional error/reset"],
            "output_payload": "ten signed int32 scores plus optional uint8 argmax class",
            "transport_status": "not frozen; AXI4-Stream + DMA is the current performance-oriented proposal",
        },
        "parameter_file": parameters_path.name,
        "parameter_file_sha256": sha256_file(parameters_path),
    }
    (OUTPUT_DIR / "candidate_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def quant_error(float_values, integer_values, fraction_bits):
        reconstructed = integer_values.astype(np.float64) * (2.0 ** -fraction_bits)
        difference = reconstructed - float_values.astype(np.float64)
        return {
            "float_abs_max": float(np.max(np.abs(float_values))),
            "integer_min": int(integer_values.min()),
            "integer_max": int(integer_values.max()),
            "saturated_count": int(np.count_nonzero(np.abs(integer_values.astype(np.int16)) == 127)),
            "max_absolute_quantization_error": float(np.max(np.abs(difference))),
            "root_mean_square_quantization_error": float(np.sqrt(np.mean(difference**2))),
        }

    range_analysis = {
        "training_relu_channel_max_float32": channel_max.tolist(),
        "training_relu_global_max_float32": activation_max,
        "activation_representable_max": 255.0 * (2.0 ** -activation_fraction_bits),
        "activation_headroom": 255.0 * (2.0 ** -activation_fraction_bits) - activation_max,
        "conv_weight_quantization": quant_error(
            conv_weight, conv_weight_q, conv_weight_fraction_bits
        ),
        "fc_weight_quantization": quant_error(fc_weight, fc_weight_q, fc_weight_fraction_bits),
        "conv_accumulator_conservative_absolute_bound_by_channel": conv_bounds.tolist(),
        "conv_accumulator_conservative_absolute_bound_max": conv_bound,
        "conv_accumulator_minimum_signed_bits_for_bound": signed_bits_for_absolute_bound(conv_bound),
        "fc_accumulator_conservative_absolute_bound_by_class": fc_bounds.tolist(),
        "fc_accumulator_conservative_absolute_bound_max": fc_bound,
        "fc_accumulator_minimum_signed_bits_for_bound": signed_bits_for_absolute_bound(fc_bound),
        "int32_headroom_confirmed": conv_bound <= np.iinfo(np.int32).max
        and fc_bound <= np.iinfo(np.int32).max,
        "parameter_storage_bytes": {
            "conv_weight_int8": int(conv_weight_q.nbytes),
            "conv_bias_int32": int(conv_bias_q.nbytes),
            "fc_weight_int8": int(fc_weight_q.nbytes),
            "fc_bias_int32": int(fc_bias_q.nbytes),
            "total": int(
                conv_weight_q.nbytes + conv_bias_q.nbytes + fc_weight_q.nbytes + fc_bias_q.nbytes
            ),
        },
        "activation_storage_bytes_per_image": {
            "input_uint8": 28 * 28,
            "conv_relu_uint8": 4 * 26 * 26,
            "pool_uint8": 4 * 13 * 13,
            "scores_int32": 10 * 4,
        },
    }
    (OUTPUT_DIR / "range_analysis.json").write_text(
        json.dumps(range_analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Demonstrate the candidate rule on the same location used by Step 5.
    inputs_uint8 = np.load(FLOAT_REF / "inputs" / "fixed_inputs_uint8.npy", allow_pickle=False)
    float_conv = np.load(
        FLOAT_REF / "activations" / "conv_output_float32.npy", allow_pickle=False
    )
    manual_float = json.loads(
        (FLOAT_REF / "manual_check" / "manual_conv.json").read_text(encoding="utf-8")
    )
    sample = int(manual_float["sample_id"])
    channel = int(manual_float["output_channel"])
    row = int(manual_float["output_row"])
    column = int(manual_float["output_column"])
    patch_q = inputs_uint8[sample, row : row + 3, column : column + 3].astype(np.int64)
    kernel_q = conv_weight_q[channel, 0].astype(np.int64)
    accumulator = int(conv_bias_q[channel]) + int((patch_q * kernel_q).sum())
    positive = max(accumulator, 0)
    rounded = (positive + (1 << (conv_requant_right_shift - 1))) >> conv_requant_right_shift
    activation_q = min(255, rounded)
    integer_example = {
        "sample_id": sample,
        "true_digit": sample,
        "output_channel": channel,
        "output_row": row,
        "output_column": column,
        "input_patch_uint8": patch_q.tolist(),
        "kernel_int8": kernel_q.tolist(),
        "bias_int32": int(conv_bias_q[channel]),
        "integer_accumulator": accumulator,
        "accumulator_fraction_bits": conv_accumulator_fraction_bits,
        "dequantized_conv_value": accumulator * (2.0 ** -conv_accumulator_fraction_bits),
        "float32_reference_conv_value": float(float_conv[sample, channel, row, column]),
        "absolute_quantization_difference": abs(
            accumulator * (2.0 ** -conv_accumulator_fraction_bits)
            - float(float_conv[sample, channel, row, column])
        ),
        "relu_requant_right_shift": conv_requant_right_shift,
        "rounding_offset": 1 << (conv_requant_right_shift - 1),
        "relu_uint8_result": activation_q,
        "relu_dequantized_value": activation_q * (2.0 ** -activation_fraction_bits),
    }
    (OUTPUT_DIR / "integer_rule_example.json").write_text(
        json.dumps(integer_example, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "input_fraction_bits": input_fraction_bits,
        "conv_weight_fraction_bits": conv_weight_fraction_bits,
        "activation_fraction_bits": activation_fraction_bits,
        "fc_weight_fraction_bits": fc_weight_fraction_bits,
        "conv_requant_right_shift": conv_requant_right_shift,
        "conv_minimum_accumulator_bits": signed_bits_for_absolute_bound(conv_bound),
        "fc_minimum_accumulator_bits": signed_bits_for_absolute_bound(fc_bound),
        "parameter_storage_bytes": range_analysis["parameter_storage_bytes"]["total"],
        "status": config["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
