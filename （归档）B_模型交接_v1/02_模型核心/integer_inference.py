"""当前交接模型的 CPU 整数标准答案。只做推理，不训练，也不调用 FPGA。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


from paths import MODEL_DIR

DEFAULT_CANDIDATE_DIR = MODEL_DIR


@dataclass
class IntegerForwardResult:
    predictions: np.ndarray
    scores_int32: np.ndarray
    conv_acc_int32: np.ndarray | None = None
    relu_uint8: np.ndarray | None = None
    pool_uint8: np.ndarray | None = None
    flatten_uint8: np.ndarray | None = None
    relu_rounded_before_saturation: np.ndarray | None = None


class IntegerDigitCNN:
    """读取定点参数，严格按整数规则计算；A 的硬件应逐层对齐本实现。"""

    def __init__(self, config: dict, parameters: dict[str, np.ndarray]):
        self.config = config
        self.conv_weight = np.ascontiguousarray(parameters["conv_weight_int8"], dtype=np.int8)
        self.conv_bias = np.ascontiguousarray(parameters["conv_bias_int32"], dtype=np.int32)
        self.fc_weight = np.ascontiguousarray(parameters["fc_weight_int8"], dtype=np.int8)
        self.fc_bias = np.ascontiguousarray(parameters["fc_bias_int32"], dtype=np.int32)
        self.right_shift = (
            config["numeric_rules"]["conv_accumulator"]["fraction_bits"]
            - config["numeric_rules"]["relu_requantized_activation"]["fraction_bits"]
        )
        if self.conv_weight.shape != (4, 1, 3, 3):
            raise ValueError("conv_weight_int8 must have shape [4,1,3,3]")
        if self.conv_bias.shape != (4,):
            raise ValueError("conv_bias_int32 must have shape [4]")
        if self.fc_weight.shape != (10, 676):
            raise ValueError("fc_weight_int8 must have shape [10,676]")
        if self.fc_bias.shape != (10,):
            raise ValueError("fc_bias_int32 must have shape [10]")
        if self.right_shift < 1:
            raise ValueError("The current runtime rule requires a positive right shift")

    @classmethod
    def from_directory(cls, directory: str | Path = DEFAULT_CANDIDATE_DIR):
        directory = Path(directory)
        config_path = directory / "candidate_config.json"
        if not config_path.is_file():
            config_path = directory / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        parameter_name = config.get("parameter_file", "参数/model_parameters_int.npz")
        archive = np.load(directory / parameter_name, allow_pickle=False)
        parameters = {name: archive[name] for name in archive.files}
        return cls(config, parameters)

    def forward(self, inputs_uint8: np.ndarray, *, return_layers: bool = False) -> IntegerForwardResult:
        """输入单张 [28,28] 或批量 [N,28,28] uint8；返回十类分数和预测数字。"""

        inputs = np.asarray(inputs_uint8)
        if inputs.ndim == 2:
            inputs = inputs[np.newaxis, :, :]
        if inputs.ndim != 3 or inputs.shape[1:] != (28, 28):
            raise ValueError("inputs_uint8 must have shape [N,28,28] or [28,28]")
        if inputs.dtype != np.uint8:
            raise TypeError("inputs_uint8 must have dtype uint8")

        # 1. 取出所有 3×3 窗口；转 int32 后做乘加，避免 uint8/int8 直接运算溢出。
        # 卷积采用互相关，不翻转权重；输出形状 [N,4,26,26]。
        windows = np.lib.stride_tricks.sliding_window_view(inputs, (3, 3), axis=(1, 2))
        conv_acc = np.einsum(
            "nhwkl,okl->nohw",
            windows.astype(np.int32, copy=False),
            self.conv_weight[:, 0].astype(np.int32),
            dtype=np.int32,
            optimize=True,
        )
        conv_acc += self.conv_bias.reshape(1, 4, 1, 1)

        # 2. ReLU 清除负数，加 128 后右移 8 位，最后饱和到 0～255。
        positive = np.maximum(conv_acc, 0)
        rounding_offset = np.int32(1 << (self.right_shift - 1))
        rounded = (positive + rounding_offset) >> self.right_shift
        relu_uint8 = np.clip(rounded, 0, 255).astype(np.uint8)

        # 3. 每通道做 2×2 最大池化，输出 [N,4,13,13]。
        count = inputs.shape[0]
        pool_uint8 = relu_uint8.reshape(count, 4, 13, 2, 13, 2).max(axis=(3, 5))
        # 4. 按通道、行、列展平；索引为 channel*169 + row*13 + column。
        flatten_uint8 = np.ascontiguousarray(pool_uint8.reshape(count, 676))
        # 5. 全连接层用 int32 累加，输出十个同尺度分数，不需要 softmax。
        scores = (
            flatten_uint8.astype(np.int32) @ self.fc_weight.astype(np.int32).T
            + self.fc_bias.reshape(1, 10)
        ).astype(np.int32)
        # 6. 相同最大分数时取较小类别编号，A 应保持同样规则。
        predictions = scores.argmax(axis=1).astype(np.int64)

        # 调试时保留中间结果；普通演示只返回分数和标签。
        if return_layers:
            return IntegerForwardResult(
                predictions=predictions,
                scores_int32=scores,
                conv_acc_int32=conv_acc,
                relu_uint8=relu_uint8,
                pool_uint8=pool_uint8,
                flatten_uint8=flatten_uint8,
                relu_rounded_before_saturation=rounded,
            )
        return IntegerForwardResult(predictions=predictions, scores_int32=scores)
