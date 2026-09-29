"""最小演示：固定输入检查或一张外部图片预测。"""
from pathlib import Path
import argparse
import sys
import numpy as np

# 运行脚本与模型核心分层存放，先把模型核心加入 Python 模块搜索路径。
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "02_模型核心"))

from integer_inference import IntegerDigitCNN
from paths import REFERENCE_FILE
from preprocess import preprocess_digit_image

def main():
    parser = argparse.ArgumentParser(description="运行 DigitCNN 整数软件参考")
    parser.add_argument("--image", type=Path, help="可选：单个完整数字图片")
    args = parser.parse_args()
    model = IntegerDigitCNN.from_directory()
    if args.image is None:
        # 固定十张按照数字 0～9 排列，用来快速确认模型加载和计算正常。
        with np.load(REFERENCE_FILE, allow_pickle=False) as reference:
            inputs = reference["fixed_inputs_uint8"]
    else:
        # 预处理返回白字黑底的 uint8[28,28]；不要再次除以 255 或 256。
        inputs = preprocess_digit_image(args.image).normalized_uint8
    result = model.forward(inputs)
    print("预测数字：", result.predictions.tolist())
    # 单张图片时显示十类分数，方便 C 调试；固定十张只显示标签，保持输出简洁。
    if args.image is not None:
        print("十类整数分数：", result.scores_int32[0].tolist())

if __name__ == "__main__":
    main()
