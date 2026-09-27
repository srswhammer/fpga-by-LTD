# Z15 OV5640 PYNQ 采集 Overlay

## 已实现功能

本工程把 OV5640 的 640×480 RGB565 图像送入 FPGA，经 AXI4-Stream 和 AXI VDMA 写入 Zynq DDR，随后由 PYNQ/Python 读取并显示。

已经在正点原子 Z15（`xc7z015clg485-2`）上完成单帧实拍验证，链路如下：

```text
OV5640 → FPGA 图像前端 → AXI4-Stream → VDMA → DDR → NumPy → RGB 显示
```

## 主要文件

- `camera_frontend.v`：摄像头初始化、RGB565 拼接及 AXI4-Stream 输出。
- `camera_overlay.xdc`：Z15 摄像头和诊断 LED 引脚约束。
- `i2c_dri.v`：摄像头 SCCB/I²C 驱动。
- `i2c_ov5640_rgb565_cfg.v`：OV5640 的 640×480 RGB565 初始化配置。
- `build_overlay.tcl`：创建 PS7、VDMA、DDR 通路并生成 Overlay。
- `camera_capture.bit`：上传到 PYNQ 的 FPGA 配置文件。
- `camera_capture.hwh`：PYNQ 识别 IP 和地址所需的硬件描述文件。

上传时必须让 `camera_capture.bit` 与 `camera_capture.hwh` 位于 Jupyter 的同一文件夹，并保持相同的主文件名。

## PYNQ 端验证结果

- Overlay 加载成功。
- 可见 IP：`vdma`、`ps7`。
- 成功取得并显示 640×480 摄像头画面。
- 旧版 PYNQ 的标准 `AxiVDMA` 驱动会因为工程中未连接中断而报错，因此当前验证代码使用 `MMIO` 直接配置 VDMA 寄存器。

## 当前限制

- 目前只验证了单帧采集，还没有整理连续预览程序。
- VDMA 状态曾显示 `0x1d890`。虽然取得的实拍画面完整可辨认，但帧同步错误仍需继续检查。
- 下一步应先稳定多次采集，再接入数独棋盘定位、透视校正和 81 格切分。

## 重新生成

使用 Vivado 2025.2，在本目录执行：

```powershell
vivado -mode batch -source build_overlay.tcl
```

成功后会在本目录得到新的 `camera_capture.bit` 和 `camera_capture.hwh`。Vivado 中间目录已由 Git 忽略。

