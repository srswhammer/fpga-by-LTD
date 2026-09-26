# 05_FPGA_Verilog：DigitCNN int8 的 Verilog 加速器

目标：Zynq-7015 + PYNQ 2.7。数值与 `02_模型核心/integer_inference.py` **逐位一致**，也就是十张固定图的池化层、十类 int32 分数和 argmax 在仿真中全部通过。

所有端口开关、位宽和寄存器地址都**先用宏定义**，集中放在 `rtl/cnn_defines.vh`。标 `[待确认]` 的地方等 Block Design 定下来后一起改。

## 目录

```text
05_FPGA_Verilog/
├── rtl/
│   ├── cnn_defines.vh      ★ 所有端口/接口/地址宏
│   ├── cnn_accel_top.v     顶层：AXIS 入 / AXIS 出 / AXI-Lite / 中断（按宏生成）
│   ├── cnn_core.v          计算核心：接收 → 卷积+ReLU+池化 → 全连接 → argmax → 输出
│   └── cnn_axil_regs.v     AXI4-Lite 寄存器
├── mem/                    参数 ROM 初始化文件（gen_mem.py 生成）
│   ├── conv_weight_int8.mem   36 × int8
│   ├── conv_bias_int32.mem    4 × int32
│   ├── fc_weight_packed.mem   676 行 × 80 bit（每行 10 类 int8，类 0 在低字节）
│   └── fc_bias_int32.mem      10 × int32
├── tb/
│   ├── tb_cnn_accel.v      testbench：十张图逐层比对 + 随机空拍/反压 + TLAST 错误帧
│   └── vectors/            来自 hardware_reference.npz 的输入与标准答案
├── scripts/
│   ├── gen_mem.py          从 parameters.npz / hardware_reference.npz 重新生成 mem
│   ├── run_sim.sh          Icarus Verilog 仿真（Linux / WSL / Git-Bash）
│   └── run_sim.bat         Icarus Verilog 仿真（Windows）
└── pynq/
    └── cnn_fpga_driver.py  板上驱动：DMA 推理、MMIO 读寄存器、自测
```

## 硬件结构

```text
DMA MM2S ──AXIS 32b──► [行拼接 28×224b RAM] ─► [4×4 取块] ─► [36 乘法 = 4 通道 × 3×3]
                                                                   │ +bias, ReLU, (+128)>>8, sat
                                                                   ▼
                                                      [2×2 取最大] ─► 池化缓存 4×169 B
                                                                   │
                                   [FC：每拍 1 激活 × 10 类 MAC，权重 BRAM 80b×676]
                                                                   ▼
                                        [argmax] ─► AXIS 32b ─► DMA S2MM（10 分数 + argmax）
                                                 └► AXI-Lite 寄存器 / 中断
```

| 阶段 | 周期 |
|---|---|
| 输入 784 B（32 bit/拍） | 196 |
| 卷积+ReLU+池化（169 位置 × 11 拍） | 1859 |
| 全连接（676 + 流水） | ~680 |
| argmax + 输出 | ~22 |
| **合计（仿真实测，含随机空拍）** | **约 2800 拍 ≈ 28 µs @100 MHz** |

资源预估（Yosys 通用 xc7 综合，Vivado 实际会不同）：约 2.4k LUT、1.4k FF、46 DSP48、5 RAMB18。XC7Z015 有 46k LUT、160 DSP，所以占用很小，后续还有空间加并行度。

## 宏（`rtl/cnn_defines.vh`）

| 宏 | 默认 | 作用 |
|---|---|---|
| `CNN_USE_AXI_LITE` | 开 | 生成 `s_axi_*` 控制口 |
| `CNN_USE_M_AXIS` | 开 | 生成 `m_axis_*` 结果流；关掉后只能用寄存器读分数 |
| `CNN_USE_IRQ` | 开 | 生成 `interrupt`（高电平） |
| `CNN_OUT_ARGMAX_WORD` | 开 | 结果流是 11 个字，第 11 个字是 argmax；关掉则是 10 个字 |
| `CNN_S_AXIS_DATA_W` | 32 | 输入流位宽，只能取 8、16 或 32（28 要能被每拍像素数整除） |
| `CNN_AXIL_ADDR_W` | 8 | AXI-Lite 地址位宽 |
| `CNN_REG_*` | 见下表 | 寄存器偏移 |
| `CNN_*_FILE` | `*.mem` | ROM 初始化文件名 |

这三种组合都已跑过仿真，全部 PASS：8 bit 全开；16 bit 不带 AXI-Lite；32 bit 不带 m_axis。

## 数据格式

- **输入**：784 字节，按行优先（列变化最快）。每拍装 `W/8` 个像素，序号小的像素放低字节，所以直接对 `uint8[784]` 的内存做 DMA 就行。最后一拍必须带 `TLAST`。
  - TLAST 提前出现：丢弃本帧，并把 STATUS.bit2 置 1。
  - 最后一拍没有 TLAST：照常计算，但同样把 bit2 置 1。
- **输出**：每拍 1 个 int32（小端），依次为 score0..score9，按宏再加一个 argmax。TLAST 在最后一个字上，`tkeep` 全为 1。
- 输入 `tkeep` 不检查，要求每一拍都是满字节。

## AXI-Lite 寄存器

| 偏移 | 名称 | 说明 |
|---|---|---|
| 0x00 | CTRL | W：bit0 软复位（自动清零）；bit1 中断使能 |
| 0x04 | STATUS | R：bit0 busy；bit1 done（粘滞）；bit2 tlast_err（粘滞）；bit3 空闲等待输入。W1C：写 1 清 bit1/bit2 |
| 0x08 | RESULT | [3:0] 预测数字 |
| 0x0C | CYCLES | 上一帧从第一拍输入到结果就绪的周期数 |
| 0x10 | FRAMES | 已完成帧数 |
| 0x14 | ID | 0x434E4E01 |
| 0x40–0x64 | SCORE0–9 | int32 分数 |

加速器**不需要 start 位**：DMA 送入数据就开始算。中断信号为 `irq_en & done`，写 STATUS 的 bit1 清除。

## 仿真

```sh
cd 05_FPGA_Verilog
python scripts/gen_mem.py        # 参数有变化时才需要
sh scripts/run_sim.sh            # 需要 iverilog
```

期望输出最后一行：`==== PASS：十张图 池化层 / 十类分数 / argmax 与 CPU 整数标准答案逐位一致 ====`

在 Vivado xsim 中仿真时：把 `tb/tb_cnn_accel.v` 设为 sim top，把 `mem/*.mem` 和 `tb/vectors/` 复制到 `<proj>.sim/sim_1/behav/xsim/`。

## Vivado 搭建步骤（建议，需一起确认）

1. 新建工程，器件选 `xc7z015clg485-1`（或你板子的速度等级）。先确认你拷贝的 PYNQ 2.7 镜像对应的板卡 preset 和 DDR 配置。
2. 添加 `rtl/*.v`、`rtl/cnn_defines.vh`（设为 Global Include）和 `mem/*.mem`（类型会自动识别为 Memory Initialization Files）。
3. 在 Block Design 中：
   - ZYNQ7 PS：Run Block Automation。打开 `M_AXI_GP0`、`S_AXI_HP0`、`FCLK_CLK0 = 100 MHz`、`IRQ_F2P`。
   - AXI DMA：取消 Scatter Gather；Width of Buffer Length Register 设为 14 及以上；MM2S 和 S2MM 的 Stream Data Width 都设为 32。
   - 右键 → Add Module… → `cnn_accel_top`，实例名 `cnn_accel_0`。
   - 连线：`axi_dma_0/M_AXIS_MM2S → cnn_accel_0/s_axis`；`cnn_accel_0/m_axis → axi_dma_0/S_AXIS_S2MM`。
   - 运行 Connection Automation：`s_axi` 和 DMA 的 `S_AXI_LITE` 接 GP0，DMA 的 `M_AXI_MM2S/S2MM` 接 HP0。
   - 用 `interrupt`、DMA 的 `mm2s_introut` 和 `s2mm_introut` 接 xlconcat，再接 `IRQ_F2P`（PYNQ 中断需要 AXI Interrupt Controller 时另加）。
4. Generate Bitstream，然后导出 `.bit` 和 `.hwh`（在 `.gen/sources_1/bd/*/hw_handoff/` 下）。两个文件同名，一起拷到板上。
5. 在板上运行：
   ```python
   from cnn_fpga_driver import CnnFpga
   acc = CnnFpga("cnn_digit.bit")
   acc.self_test("hardware_reference.npz")
   ```

## 待确认清单（对应接口说明.md 第 4 节）

- [ ] AXIS 位宽是 32 还是 8 或 16？每拍的字节排列按"低字节 = 先到像素"是否 OK？
- [ ] 结果流用 10 个字还是 11 个字？还是不要结果流，只用 AXI-Lite 读？
- [ ] 寄存器地址表、ID 值
- [ ] 中断是否使用；PYNQ 端用轮询还是 `asyncio` 中断
- [ ] 参数目前用 ROM 初始化（改参数要重新综合）。是否需要运行时经 AXI 加载？
- [ ] 板卡 preset、FCLK 频率、DMA / IP 实例名（在 `pynq/cnn_fpga_driver.py` 顶部）
- [ ] 是否需要读出中间层调试数据（现在仿真里可以直接看 `pool_b0..3`）

## 可继续优化

- 卷积与全连接可以并行流水：每产生一个池化位置就立即做 4×10 MAC，这样能省掉约 680 拍。
- 全连接可以 4 通道并行（每拍 40 MAC）：676 拍降到 169 拍。
- 卷积可以按两行一组复用 4×4 块，每个位置减少 2–3 拍取数。
