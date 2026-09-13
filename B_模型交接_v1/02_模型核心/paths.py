"""交接包统一路径表。目录调整时只需先核对这里。"""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PACKAGE_ROOT / "02_模型核心"
REFERENCE_FILE = PACKAGE_ROOT / "03_FPGA交接" / "hardware_reference.npz"
EXAMPLE_FILE = PACKAGE_ROOT / "01_运行演示" / "示例数字1.png"
GENERATED_FPGA_DIR = PACKAGE_ROOT / "generated_fpga"
