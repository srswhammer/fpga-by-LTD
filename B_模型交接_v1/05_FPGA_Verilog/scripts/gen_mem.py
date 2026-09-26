"""从唯一参数 parameters.npz 和 hardware_reference.npz 生成 Verilog 用的 .mem 文件。

生成内容：
  mem/conv_weight_int8.mem     36 行，每行 1 个 int8（2 位十六进制补码），索引 = c*9 + kr*3 + kc
  mem/conv_bias_int32.mem      4 行，每行 1 个 int32（8 位十六进制补码）
  mem/fc_weight_packed.mem     676 行，每行 80 bit = 10 个 int8；bit[k*8+:8] = 第 k 类权重
                               行号 = 展平索引 channel*169 + row*13 + column
  mem/fc_bias_int32.mem        10 行
  tb/vectors/*.mem             testbench 用固定十张输入和逐层标准答案

用法（在交接根目录）：python "05_FPGA_Verilog/scripts/gen_mem.py"
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FPGA_DIR = HERE.parent                      # 05_FPGA_Verilog
ROOT = FPGA_DIR.parent                      # 交接根目录
PARAM = ROOT / "02_模型核心" / "parameters.npz"
REF = ROOT / "03_FPGA交接" / "hardware_reference.npz"
MEM_DIR = FPGA_DIR / "mem"
VEC_DIR = FPGA_DIR / "tb" / "vectors"


def hexs(v, bits):
    return f"{int(v) & ((1 << bits) - 1):0{bits // 4}x}"


def write_lines(path, lines):
    path.write_text("".join(l + "\n" for l in lines), encoding="ascii")
    print(f"  {path.relative_to(FPGA_DIR)}  ({len(lines)} 行)")


def main():
    MEM_DIR.mkdir(exist_ok=True)
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    p = np.load(PARAM, allow_pickle=False)
    cw, cb = p["conv_weight_int8"], p["conv_bias_int32"]
    fw, fb = p["fc_weight_int8"], p["fc_bias_int32"]
    assert cw.shape == (4, 1, 3, 3) and fw.shape == (10, 676)

    print("模型参数：")
    write_lines(MEM_DIR / "conv_weight_int8.mem", [hexs(v, 8) for v in cw.reshape(-1)])
    write_lines(MEM_DIR / "conv_bias_int32.mem", [hexs(v, 32) for v in cb])
    # 每个展平索引一行，10 类权重打包，类别 0 在最低字节
    packed = []
    for i in range(676):
        word = "".join(hexs(fw[k, i], 8) for k in range(9, -1, -1))
        packed.append(word)
    write_lines(MEM_DIR / "fc_weight_packed.mem", packed)
    write_lines(MEM_DIR / "fc_bias_int32.mem", [hexs(v, 32) for v in fb])

    r = np.load(REF, allow_pickle=False)
    print("仿真向量：")
    write_lines(VEC_DIR / "inputs_u8.mem", [hexs(v, 8) for v in r["fixed_inputs_uint8"].reshape(-1)])
    write_lines(VEC_DIR / "pool_u8.mem", [hexs(v, 8) for v in r["flatten_uint8"].reshape(-1)])
    write_lines(VEC_DIR / "scores_i32.mem", [hexs(v, 32) for v in r["scores_int32"].reshape(-1)])
    write_lines(VEC_DIR / "pred.mem", [hexs(v, 8) for v in r["predictions_int64"]])


if __name__ == "__main__":
    main()
