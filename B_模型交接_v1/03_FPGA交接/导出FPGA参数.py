"""按 A 的硬件方案，现场从唯一 NPZ 参数生成 MEM、BIN 和布局说明。"""
from pathlib import Path
import json
import sys
import numpy as np

# 本脚本只生成硬件格式副本，唯一参数仍在 02_模型核心/parameters.npz。
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "02_模型核心"))

from paths import MODEL_DIR, GENERATED_FPGA_DIR

ORDER = ("conv_weight_int8", "conv_bias_int32", "fc_weight_int8", "fc_bias_int32")

def write_mem(path: Path, values: np.ndarray) -> None:
    # 负数先转为对应位宽的补码，再按每行一个整数写十六进制。
    bits = values.dtype.itemsize * 8
    mask = (1 << bits) - 1
    width = bits // 4
    path.write_text("".join(f"{int(v) & mask:0{width}x}\n" for v in values.reshape(-1)), encoding="ascii")

def main():
    GENERATED_FPGA_DIR.mkdir(exist_ok=True)
    with np.load(MODEL_DIR / "parameters.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ORDER}
    offset = 0
    layout = []
    binary_parts = []
    for name in ORDER:
        value = np.ascontiguousarray(arrays[name])
        if value.dtype == np.int32:
            # BIN 明确使用 little-endian int32，避免不同电脑端序不一致。
            raw = value.astype("<i4", copy=False).tobytes(order="C")
        else:
            raw = value.tobytes(order="C")
        write_mem(GENERATED_FPGA_DIR / f"{name}.mem", value)
        layout.append({"name": name, "offset": offset, "bytes": len(raw), "shape": list(value.shape)})
        binary_parts.append(raw)
        offset += len(raw)
    (GENERATED_FPGA_DIR / "model_parameters.bin").write_bytes(b"".join(binary_parts))
    (GENERATED_FPGA_DIR / "binary_layout.json").write_text(
        json.dumps({"endianness": "little for int32", "total_bytes": offset, "segments": layout},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FPGA 参数已生成：", GENERATED_FPGA_DIR)
    print("总原始字节数：", offset)

if __name__ == "__main__":
    main()
