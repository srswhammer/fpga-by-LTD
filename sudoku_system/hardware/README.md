# hardware/ — 硬件源码

| 模块 | 源码在哪 | 编译好的 bit |
|---|---|---|
| LCD 显示（红色版） | `hardware/lcd/`（完整 Verilog、约束、构建脚本、仿真、报告） | `overlays/lcd.bit` |
| 硬件 CNN + 求解器 | **不在本文件夹**。`cnn_solver.hwh` 里记录的权重路径是 `C:/Users/hammer/Desktop/final/mem/`，Vivado 工程应在组长的 `final` 文件夹 | `overlays/cnn_solver.bit` |
| 摄像头 + VDMA | **不在本文件夹**（`camera_frontend` 的源码在摄像头工程里） | `overlays/camera.bit` |

## LCD 源码说明

- 来自 `dyx_lcd_red.zip`，是红色版：原题黑色，补入数字纯红（`sudoku_pixels.v` 里的 `24'hff0000`）。
  旧文件夹 `dyx_lcd/` 里是蓝色版（`24'h0055cc`），其余文件相同。
- 器件 `xc7z015clg485-2`，系统时钟 50 MHz（引脚 Y14），像素时钟 25 MHz；屏幕固定为正点原子 4.3 寸 800×480（ID 4384）。
- 数据入口是 PL UART（AA11/AB11，115200 8N1）：`55 AA` + 81 字节 + 校验和，正确回 `06`，错误或超时回 `15`。
- 上电默认画面在 `rtl/board_init.vh`（2026-10-10 现场识别结果）。
- 重新编译：在 `hardware/lcd/` 里运行 `vivado -mode batch -source build.tcl`，输出 `build/lcd_display.bit`，
  改名为 `lcd.bit` 放进 `overlays/`。
- 仿真：`tb_bridge.v` 测串口协议（整帧更新、ACK、校验、保留位、超时），`tb_pixels.v` 测网格和颜色。
