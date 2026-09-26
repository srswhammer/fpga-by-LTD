"""从 hardware_reference.npz 的固定十张输入生成 Verilog 仿真用 .mem 文件。

只读取交接包里已验证过的标准答案（不重新跑模型、不改交接目录），把
$readmemh 能直接加载的十六进制文件放到 module/sim/ 下，供
module/sim/tb_cnn_top.v 做十张图片的逐张自检。
"""
from pathlib import Path
import sys

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = MODULE_ROOT.parent / "B_模型交接_v1"
sys.path.insert(0, str(PACKAGE_ROOT / "02_模型核心"))

from paths import REFERENCE_FILE  # noqa: E402

SIM_DIR = MODULE_ROOT / "sim"


def write_hex(path: Path, values: np.ndarray) -> None:
    bits = values.dtype.itemsize * 8
    mask = (1 << bits) - 1
    width = bits // 4
    path.write_text(
        "".join(f"{int(v) & mask:0{width}x}\n" for v in values.reshape(-1)),
        encoding="ascii",
    )


def main() -> None:
    if not REFERENCE_FILE.is_file():
        raise SystemExit(f"未找到标准答案文件：{REFERENCE_FILE}")

    SIM_DIR.mkdir(exist_ok=True)
    with np.load(REFERENCE_FILE, allow_pickle=False) as archive:
        inputs = archive["fixed_inputs_uint8"]       # [10,28,28] uint8
        labels = archive["labels_int64"]              # [10]
        scores = archive["scores_int32"]               # [10,10] int32
        predictions = archive["predictions_int64"]     # [10]

    count = inputs.shape[0]
    write_hex(SIM_DIR / "tb_images.mem", inputs.astype(np.uint8))
    write_hex(SIM_DIR / "tb_scores.mem", scores.astype(np.int32))
    write_hex(SIM_DIR / "tb_predictions.mem", predictions.astype(np.uint8))
    write_hex(SIM_DIR / "tb_labels.mem", labels.astype(np.uint8))

    print(f"已生成 {count} 张固定测试图片的仿真数据到 {SIM_DIR}")
    print("predictions:", predictions.tolist())
    print("labels:     ", labels.tolist())


if __name__ == "__main__":
    main()
