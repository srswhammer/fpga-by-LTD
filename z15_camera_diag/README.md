# Z15 OV5640 硬件诊断工程

## 用途

这个工程用于在不启动 PYNQ 图像采集的情况下，快速检查 Z15 开发板上的 OV5640 摄像头是否能够初始化，以及摄像头是否输出像素时钟。

适用硬件：正点原子 Z15，FPGA 型号 `xc7z015clg485-2`。

## 指示灯含义

- `PL_LED0 / led_init` 常亮：OV5640 寄存器初始化完成。
- `PL_LED1 / led_pclk` 闪烁：检测到摄像头像素时钟。

实测结果：两项均正常，说明摄像头、排线、FMC 接口、SCCB 配置和像素时钟链路能够工作。

## 主要文件

- `camera_diag_top.v`：诊断顶层和指示灯逻辑。
- `camera_diag.xdc`：Z15 摄像头及 LED 引脚约束。
- `i2c_dri.v`：摄像头 SCCB/I²C 驱动。
- `i2c_ov5640_rgb565_cfg.v`：OV5640 的 640×480 RGB565 初始化配置。
- `build.tcl`：自动创建 Vivado 工程并生成 bitstream。
- `camera_diag_top.bit`：已经生成并通过上板验证的诊断 bitstream。

## 重新生成

使用 Vivado 2025.2，在本目录执行批处理脚本：

```powershell
vivado -mode batch -source build.tcl
```

生成的最终文件会复制到本目录的 `camera_diag_top.bit`。Vivado 中间目录已由 Git 忽略。

