"""精简交接验收：检查文件、参数格式和固定十张的逐层整数结果。"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

# 从独立验证目录调用模型核心，避免复制推理代码。
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "02_模型核心"))

from integer_inference import IntegerDigitCNN
from paths import MODEL_DIR, REFERENCE_FILE

EXPECTED_PARAMETER_SHA256 = "9cac4267a3dd78235c17ba2c9bc074216eb37fb7d592cd1f3214ea4200deeded"
EXPECTED_REFERENCE_SHA256 = "afff2f0e850c73755709cc331d6b4552cc841593d1339c213553883d08682cea"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parameter_file = MODEL_DIR / "parameters.npz"
    if sha256(parameter_file) != EXPECTED_PARAMETER_SHA256:
        raise ValueError("参数文件损坏或被改动")
    if sha256(REFERENCE_FILE) != EXPECTED_REFERENCE_SHA256:
        raise ValueError("硬件标准答案损坏或被改动")

    # 参数形状是 PS 软件、A 的硬件和模型结构共同遵守的合同。
    with np.load(parameter_file, allow_pickle=False) as p:
        expected_shapes = {
            "conv_weight_int8": ((4, 1, 3, 3), np.dtype("int8")),
            "conv_bias_int32": ((4,), np.dtype("int32")),
            "fc_weight_int8": ((10, 676), np.dtype("int8")),
            "fc_bias_int32": ((10,), np.dtype("int32")),
        }
        for name, (shape, dtype) in expected_shapes.items():
            if name not in p or p[name].shape != shape or p[name].dtype != dtype:
                raise ValueError("参数格式错误：" + name)

    model = IntegerDigitCNN.from_directory()
    with np.load(REFERENCE_FILE, allow_pickle=False) as reference:
        result = model.forward(reference["fixed_inputs_uint8"], return_layers=True)
        checks = {
            "conv_acc_int32": result.conv_acc_int32,
            "relu_uint8": result.relu_uint8,
            "pool_uint8": result.pool_uint8,
            "flatten_uint8": result.flatten_uint8,
            "scores_int32": result.scores_int32,
            "predictions_int64": result.predictions,
        }
        for name, actual in checks.items():
            if not np.array_equal(actual, reference[name]):
                raise ValueError("逐层结果不一致：" + name)
        if not np.array_equal(result.predictions, reference["labels_int64"]):
            raise ValueError("固定十张没有全部正确识别")

    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    print("验收通过：参数完整、逐层结果一致、固定十张识别为 0～9")
    print("模型版本：", config["format_version"])
    print("注意：该结论是 CPU 整数参考验证，不是 FPGA 上板验证")

if __name__ == "__main__":
    main()
