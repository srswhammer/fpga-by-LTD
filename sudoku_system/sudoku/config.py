"""所有可调设置集中在这里：要改路径、网络、超时、开关，只改这一个文件。

放进来的原则：换个环境或换种用法就可能要改的东西。
故意没放进来的：
  - 寄存器偏移、IP 默认地址、图像尺寸 640×480（在 camera.py / cnn.py / solver.py 里）
    这些由 bit 文件决定，在这里改了也不会改变硬件，放进来反而误导。
  - 切格和预处理的算法参数（grid.py / preprocess.py 里）
    已经上板验证过，随便改会直接影响识别结果。

临时改某一项（只对当前内核有效，不改文件）：
    from sudoku import config
    config.BRIDGE_URL = 'http://192.168.2.5:8765'
"""
from pathlib import Path

# ============ 路径 ============
ROOT = Path(__file__).resolve().parent.parent       # sudoku_system/
OVERLAYS = ROOT / 'overlays'                         # 三个 bit 放这里
RESULTS = ROOT / 'results'                           # 每次运行的记录放这里

CAMERA_BIT = OVERLAYS / 'camera.bit'                 # 摄像头 + VDMA
CNN_BIT = OVERLAYS / 'cnn_solver.bit'                # 硬件 CNN + 求解器
LCD_BIT = OVERLAYS / 'lcd.bit'                       # LCD 显示（纯 PL，没有 hwh）

# 开跑前检查这些文件都在；.bit 和同名 .hwh 必须成对
REQUIRED_FILES = [CAMERA_BIT, CAMERA_BIT.with_suffix('.hwh'),
                  CNN_BIT, CNN_BIT.with_suffix('.hwh'),
                  LCD_BIT]

# ============ 电脑桥（LCD 数据经电脑的 USB 串口发给 PL UART）============
# 电脑有线 IP 192.168.2.1，板子 192.168.2.99；电脑端运行 lcd_bridge.py --port COM3
BRIDGE_URL = 'http://192.168.2.1:8765'
BRIDGE_TIMEOUT = 10          # 等电脑桥回复的最长时间（秒）
LCD_ACK = 0x06               # LCD 收到正确数据帧后回的字节（出错回 0x15）
LCD_SETTLE = 0.3             # 加载 lcd.bit 后等多久再发数据（秒）

# ============ 摄像头 ============
CAPTURE_WAIT = 1.0           # 启动 VDMA 后等多久再停止取帧（秒）；验证过的值是 1.0
SHOW_PHOTO = True            # 拍完是否在 Notebook 里显示照片

# ============ 识别 / 求解 ============
CELL_POLARITY = 'auto'       # 单格黑白判断：'auto' / 'dark_on_light' / 'light_on_dark'
SOLVER_TIMEOUT = 1.0         # 等求解器完成的最长时间（秒）

# ============ 结果保存 ============
RUN_PREFIX = 'live'          # 结果目录名：results/live_xxxxxxxx/
SAVE_DEBUG = True            # 是否保存中间图（processing/ 和 cells/）；
                             # capture.png 和 lcd_result 总是保存
