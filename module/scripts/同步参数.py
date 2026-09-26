"""把 B_模型交接_v1/generated_fpga/ 的四个参数 .mem 复制到 module/params/。

module/rtl/*.v 里 $readmemh 的默认路径指向 module/params/（纯 ASCII 路径），
不直接引用 B_模型交接_v1（含中文字符）下的路径——部分仿真/综合工具的
$readmemh 处理含中文的相对路径时会读取失败（Icarus Verilog 实测复现）。

模型更新后，先运行 B_模型交接_v1/03_FPGA交接/导出FPGA参数.py 重新生成
generated_fpga/ 下的 .mem，再运行本脚本同步到 module/params/。
"""
from pathlib import Path
import shutil
import sys

MODULE_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = MODULE_ROOT.parent / "B_模型交接_v1"
sys.path.insert(0, str(PACKAGE_ROOT / "02_模型核心"))

from paths import GENERATED_FPGA_DIR  # noqa: E402

FILES = (
    "conv_weight_int8.mem",
    "conv_bias_int32.mem",
    "fc_weight_int8.mem",
    "fc_bias_int32.mem",
)


def main() -> None:
    if not GENERATED_FPGA_DIR.is_dir():
        raise SystemExit(
            f"未找到 {GENERATED_FPGA_DIR}，请先运行 "
            "B_模型交接_v1/03_FPGA交接/导出FPGA参数.py"
        )

    params_dir = MODULE_ROOT / "params"
    params_dir.mkdir(exist_ok=True)

    for name in FILES:
        src = GENERATED_FPGA_DIR / name
        if not src.is_file():
            raise SystemExit(f"缺少 {src}，请先重新运行导出脚本")
        shutil.copyfile(src, params_dir / name)

    print(f"已同步 {len(FILES)} 个参数文件到 {params_dir}")


if __name__ == "__main__":
    main()
