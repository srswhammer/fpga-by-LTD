# sudoku_system — 摄像头数独识别求解（PYNQ / Zynq-7015）

拍照 → 找棋盘、切 81 格 → **硬件 CNN** 识别 → **硬件求解器** → LCD 显示答案（原题黑色、补入红色）。

整个项目只有这一个文件夹、一个入口。它由原来分散的 `cnn_sudoku/`、`dyx_camera/`、`dyx_lcd_red/`、`dyx_one_click/` 整理而来，硬件（三个 bit）没有改动。

> **状态（2026-10-12）**：各模块已在电脑上离线验证（切格与原程序逐像素一致、LCD 数据与原来一致、电脑桥与板端联调模拟通过），**整理后的版本尚未上板运行**。上板请按下面"第一次上板"的顺序验证。

---

## 快速开始

1. 把整个 `sudoku_system/` 拷到板子的 `/home/xilinx/jupyter_notebooks/` 下
2. 电脑上启动电脑桥：`python pc/lcd_bridge.py --port COM3`（详见 `pc/README.md`）
3. 板子上打开 `run.ipynb`，运行"运行"那一格

每次运行的照片、81 格、LCD 数据保存在 `results/live_xxxxxxxx/`。

## 第一次上板（按顺序，前一步通过再做下一步）

| 顺序 | 运行 | 需要 | 通过标准 |
|---|---|---|---|
| 1 | `tests/test_offline.py` | 只要板子 | 全部 PASS（切格必须逐像素一致） |
| 2 | `tests/test_board.py` | 加载 cnn_solver.bit | 最后一行 PASS |
| 3 | `tools/1_camera.ipynb` | 摄像头 | 照片正常，81 格切得对 |
| 4 | `tools/2_solve.ipynb` | — | 显示"解出"，原题与纸面一致 |
| 5 | `tools/3_lcd.ipynb` | 电脑桥、PL UART | 收到 `{'ack': 6}`，屏幕显示该盘面 |
| 6 | `run.ipynb` | 全部 | "完成：LCD已确认接收本次答案" |

`test_*.py` 的运行方法：在 Jupyter 里打开 `tests/`，新建 Notebook，执行 `%run test_offline.py`。

---

## 文件夹结构

```
sudoku_system/
├── README.md                本文件
├── run.ipynb                ★ 唯一入口：一键运行
│
├── overlays/                三个 bit（.bit 和同名 .hwh 必须成对）
│   ├── camera.bit / .hwh        摄像头 + VDMA          ← dyx_camera/camera_capture.*
│   ├── cnn_solver.bit / .hwh    硬件 CNN + 求解器      ← cnn_sudoku/system_wrapper.*
│   └── lcd.bit                  LCD 显示（红色版）     ← dyx_lcd_red/lcd_display.bit
│
├── sudoku/                  Python 包（所有板上代码）
│   ├── config.py                ★ 所有可调设置：路径、电脑桥地址、超时、开关
│   ├── camera.py                拍照
│   ├── grid.py                  找棋盘、切 81 格
│   ├── preprocess.py            单格预处理成 28×28
│   ├── cnn.py                   写 BRAM、启动 CNN
│   ├── solver.py                81 格送入 CNN + 求解器，读回答案
│   ├── lcd.py                   校验答案、打包 81 字节、经电脑桥发送
│   └── pipeline.py              把以上串成一次运行
│
├── tools/                   分步调试（结果都在 results/manual/）
│   ├── 1_camera.ipynb           拍照切格
│   ├── 2_solve.ipynb            硬件识别 + 求解（可用样例代替拍照）
│   └── 3_lcd.ipynb              LCD 显示
│
├── tests/
│   ├── test_offline.py          不加载 bit 的自检（板子、电脑都能跑）
│   ├── test_board.py            硬件 CNN + 求解器上板自检（组长原版）
│   └── data/
│       ├── joint_cells.mem / joint_puzzle.txt / joint_solution.txt   仿真同款数据
│       ├── sample_capture.png       一张真实拍摄的照片
│       ├── sample_cells/            板子当时从这张照片切出的 81 格
│       └── sample_lcd_result.json   这张照片对应的题目、答案和 81 字节
│
├── pc/                      电脑端（不用拷到板子）
│   ├── lcd_bridge.py            电脑桥：HTTP → 串口 → LCD
│   ├── send_board.ps1           电脑直接给 LCD 发一份结果
│   └── README.md                网络、防火墙、常见问题
│
├── hardware/                硬件源码
│   ├── README.md                CNN、摄像头的 Vivado 源码不在这里，说明在哪
│   └── lcd/                     红色版 LCD：rtl/、约束、build.tcl、仿真、报告
│
├── reference/int8_model/    软件 INT8 参考模型（只做对照，不在上板流程里）
│
└── results/                 每次运行的记录
    ├── live_xxxxxxxx/           run.ipynb 生成
    └── manual/                  tools/ 生成
```

各文件夹里的 `.gitkeep` 是建空文件夹时留的占位文件，可以删除。

---

## 运行流程

```
run()                                     sudoku/pipeline.py
 ├─ check_files()       三个 bit 都在？
 ├─ lcd.check_bridge()  电脑桥开着？（没开就不拍照）
 ├─ 1/3 capture()       加载 camera.bit → VDMA 抓一帧 → results/…/capture.png
 │      split_cells()   找棋盘 → 透视矫正 → 光照校正 → 81 格 → cells/
 ├─ 2/3 solve()         预处理 28×28 → 加载 cnn_solver.bit → 逐格送 CNN → 求解器
 └─ 3/3 display()       校验答案 → lcd_result.json → 加载 lcd.bit → 电脑桥 → 串口 → 等 ACK
```

三个 bit 依次加载，加载下一个会替换上一个；任何一步失败都会直接报错停下，不会用旧数据顶替。

## 设置（`sudoku/config.py`）

| 设置 | 默认 | 说明 |
|---|---|---|
| `BRIDGE_URL` | `http://192.168.2.1:8765` | 电脑桥地址 |
| `BRIDGE_TIMEOUT` | 10 秒 | 等电脑桥回复 |
| `LCD_SETTLE` | 0.3 秒 | 加载 lcd.bit 后等多久再发 |
| `CAPTURE_WAIT` | 1.0 秒 | 取帧等待（验证过的值） |
| `SHOW_PHOTO` | True | 拍完是否显示照片 |
| `CELL_POLARITY` | `'auto'` | 单格黑白判断 |
| `SOLVER_TIMEOUT` | 1.0 秒 | 等求解器完成 |
| `SAVE_DEBUG` | True | 是否保存中间图（照片和 lcd_result 总是保存） |
| `RUN_PREFIX` | `'live'` | 结果目录名前缀 |

临时修改（只对当前内核有效）：`from sudoku import config; config.SAVE_DEBUG = False`。

寄存器偏移、IP 地址、640×480 分辨率由 bit 决定，切格和预处理的算法参数已经验证过，这些故意不放进 config。

---

## 与原文件的对应关系

| 新文件 | 原文件 | 改了什么 |
|---|---|---|
| `sudoku/cnn.py` | `cnn_sudoku/cnn_board.py` | 一行导入 |
| `sudoku/solver.py` | `cnn_sudoku/sudoku_board.py` | 一行导入；求解器默认地址改为实际的 `0x40002000` |
| `sudoku/preprocess.py` | `cnn_sudoku/preprocess.py` | 未改（`dyx_camera` 里那份与它逐字节相同，已合并） |
| `sudoku/camera.py` | `dyx_one_click/camera_runtime.py` | 改成 `Camera` 类，用完自动停 VDMA、释放缓冲区；寄存器操作不变 |
| `sudoku/grid.py` | `dyx_camera/processing/recognize.py` | 只保留切格部分，去掉对软件模型的依赖；算法不变 |
| `sudoku/lcd.py` | `dyx_one_click/run_all.py` | `request_json()`、`validate_result()` 原样；LCD 那一步拆成函数 |
| `sudoku/pipeline.py` | `dyx_one_click/run_all.py` 的 `main()` | 流程不变；去掉 `os.chdir`、`sys.path`、`sys.modules`、`exec` |
| `tests/test_board.py` | `cnn_sudoku/test_board.py` | 导入和路径 |
| `pc/send_board.ps1` | `dyx_lcd/send_board.ps1` | 默认结果文件路径 |
| `pc/lcd_bridge.py` | （原版不在 jupyter_notebooks 里） | 按原接口重写 |
| `hardware/lcd/` | `dyx_lcd_red.zip` | 未改 |
| `reference/int8_model/` | `dyx_camera/processing/` | 未改 |

## 注意

- 求解出合法答案**不代表识别全对**，请对照照片核对原题（`results/…/processing/01_detected_board.png`）。
- 运行任何会加载 bit 的 Notebook 前，先关闭其他用到硬件的 Notebook 内核。
- 原来的 `cnn_sudoku/`、`dyx_camera/`、`dyx_one_click/` 里有些文件已经移进本文件夹，旧入口（`run_images.py`、`run_all.py`、`camera_export.ipynb`）已不能直接运行；以本文件夹为准。
