# DigitCNN FPGA 加速项目框架与软硬转换指南

2026-09-23

## 1. 项目总览

分工方案：摄像头采集、图片预处理和界面留在 PS 端（ARM + Python），卷积到 argmax 的整条推理放进 PL（Verilog）。两边只交换 784 字节输入和 44 字节输出。硬件结果必须和 `integer_inference.py` 逐位一致；100 MHz 下 PL 端约 2800 拍，也就是约 28 µs 一帧。

```text
OV5640 / 图片 → PS: preprocess (28×28 uint8) → DMA MM2S (784 B)
    → PL: CNN 核 (conv→relu→pool→fc→argmax)
    → DMA S2MM (44 B) → PS: 显示结果
PS: MMIO ⇄ PL: CNN 核（状态 / 寄存器）
```

箭头方向就是数据方向：只有 CNN 核在 PL 里，其余都在 PS。

| 层 | 软件形状 | 运算量（乘加） | 放在哪 |
| --- | --- | --- | --- |
| 预处理 | 任意图 → [28,28] | 分支多、不规则 | PS |
| 卷积 3×3 | [4,26,26] | 24,336 | PL |
| ReLU/重量化 | [4,26,26] | 2,704 次比较移位 | PL |
| 池化 2×2 | [4,13,13] | 2,028 次比较 | PL |
| 全连接 | [10] | 6,760 | PL |
| argmax | 标量 | 9 次比较 | PL |

选择标准：规则、重复、定点的运算放进 PL；分支多、要调参、会改动的逻辑留在 PS。

## 2. 项目目录框架

按“软件标准 → 硬件实现 → 验证 → 上板”四层组织；01–04 是已有的交接内容，05 是硬件部分，06 是还要建的 Vivado 工程。

```text
B_模型交接_v1/
├── 02_模型核心/                软件标准答案（硬件对齐的唯一依据）
│   ├── config.json            位宽、小数位、舍入规则
│   ├── parameters.npz         权重偏置
│   └── integer_inference.py   逐层整数推理
├── 03_FPGA交接/
│   └── hardware_reference.npz 十张图 + 逐层中间结果
├── 05_FPGA_Verilog/              硬件部分
│   ├── scripts/gen_mem.py     ① 参数 npz → .mem（软→硬的数据桥）
│   ├── mem/                   ② ROM 初始化文件
│   ├── rtl/
│   │   ├── cnn_defines.vh     ③ 所有端口/地址宏
│   │   ├── cnn_core.v         ④ 计算核（模块 1–6）
│   │   ├── cnn_axil_regs.v    ⑤ 寄存器（模块 7）
│   │   └── cnn_accel_top.v    ⑥ 顶层封装
│   ├── tb/                    ⑦ 仿真，向量来自 hardware_reference
│   └── pynq/cnn_fpga_driver.py ⑧ 板上驱动
└── 06_Vivado工程/              待建：Block Design、bit、hwh
```

记住一条依赖链：改 `parameters.npz` → 重跑 `gen_mem.py` → 重新仿真 → 重新综合。其他文件没有隐含依赖。

## 3. 通用软硬转换方法：五步法

每一层都按同样五步把 Python 变成 Verilog。后面每个模块小节都按这五步写。

| 步骤 | 要回答的问题 | 软件里的样子 | 硬件里的样子 |
| --- | --- | --- | --- |
| ① 定点化 | 每个数多少位、有没有符号、怎么舍入 | `np.int32`、`np.clip` | `reg signed [31:0]`、比较器 |
| ② 循环展开 | 哪几层循环并行、哪几层用计数器轮流 | `for` / `einsum` 一次算完 | N 个乘法器 + 状态机计数器 |
| ③ 存储映射 | 数组放寄存器、LUTRAM 还是 BRAM，位宽多少 | `ndarray` 任意索引 | `reg [W-1:0] mem [0:D-1]`，每拍只能读 1–2 个地址 |
| ④ 时序流水 | 每拍做多少事、哪里插寄存器 | 没有时钟概念 | valid 信号跟着数据走，每级 1 拍 |
| ⑤ 接口 | 数据怎么进出 | 函数参数 / 返回值 | AXIS（valid/ready/last）、AXI-Lite |

三条硬规则：

- **位宽要先算再写**：uint8×int8 乘积是 17 位有符号；9 项相加再加 int32 偏置，用 32 位。
- **有符号要显式写**：uint8 要先补 0 再转有符号，写成 `$signed({1'b0, x})`，否则 200 会被当成 -56。
- **每一层都要能单独比对**：硬件中间结果存在能读出的地方（如池化缓存），跟 `hardware_reference.npz` 逐层对。

## 4. 模块 0：预处理（留在 PS，不转硬件）

结论：`preprocess.py` 不转成 Verilog。它包含灰度化、背景判断、裁剪、缩放和重心居中，分支多、还要按实拍调参；且每帧只跑一次，ARM 上几毫秒就够。

这一层要做的是“定契约”，不是转换：

- 输出固定为 `uint8[28,28]`，白字黑底，行优先。这 784 字节原样交给 DMA，不再除以 255。
- 硬件永远不关心图来自哪里，所以换摄像头、改预处理都不用重新综合。
- 板上调试时，同一份 784 字节同时送 CPU 版 `model.forward()` 和 FPGA，对比分数。

以后想快一点才考虑把它下放：最适合下放的是灰度化和缩放（规则运算），重心居中和背景判断建议一直留在 PS。

## 5. 模块 1：输入接收（numpy 数组 → AXIS + 行存储）

这一步把一个“谁都能随便索引的数组”变成“每拍流进 4 个字节”的数据流，再存成卷积好读的形状。

**软件：** `inputs = np.asarray(img)  # [28,28] uint8`，整个数组一次就在内存里。

| 步骤 | 这一层的做法 |
| --- | --- |
| ① 定点化 | 像素就是 uint8，不用改 |
| ② 展开 | 32 位 AXIS 每拍 4 个像素，7 拍一行，196 拍一帧 |
| ③ 存储 | 按行存：`reg [223:0] img_row [0:27]`，一个地址 = 整行 28 像素。这样卷积每拍能拿到一整行 |
| ④ 时序 | 用 `row_buf` 先拼完 7 拍，第 7 拍整行写入 |
| ⑤ 接口 | AXIS：`tvalid && tready` 才算收到，最后一拍 `tlast=1` |

**字节序要对齐内存：** DMA 从内存低地址读，地址小的字节落在 `tdata[7:0]`。所以第 n 个像素在这一拍的 `[(n%4)*8 +: 8]`，在行里的位置就是 `[col*8 +: 8]`。

```verilog
// 拼行：把新来的 32 位放到当前列位置
row_next = row_buf;
row_next[ld_col*8 +: 32] = s_axis_tdata;
if (beat && ld_col == 24) img_row[ld_row] <= row_next;   // 行尾整行写入
```

**自己写时要注意：** TLAST 提前或缺失要有处理。否则 DMA 一次出错，后面每帧都会错位。

## 6. 模块 2：卷积（einsum → 4×4 取块 + 36 个乘法器）

这一层是转换的核心：软件一行 `einsum` 算完 24,336 次乘加，硬件要决定每拍算哪一部分。当前方案每拍算 36 个（4 通道 × 3×3）。

**软件：**

```python
windows = sliding_window_view(inputs, (3, 3))            # [26,26,3,3]
conv_acc = einsum("hwkl,okl->ohw", windows, w) + bias    # [4,26,26] int32
```

**先把 einsum 改写成循环**（这是转硬件的关键一步）：

```python
for oh in range(26):                  # ← 计数器轮流
  for ow in range(26):                # ← 计数器轮流
    for c in range(4):                # ← 并行：4 套硬件
      acc = bias[c]
      for kr in range(3):             # ← 并行
        for kc in range(3):           # ← 并行
          acc += img[oh+kr][ow+kc] * w[c][kr][kc]
```

标“并行”的内三层展开成 36 个乘法器，外两层变成状态机计数器。并行度就是这样选的：展开越多越快，但 DSP 和 LUT 越多。

| 步骤 | 这一层的做法 |
| --- | --- |
| ① 定点化 | uint8 × int8 = 17 位有符号；偏置 int32，F=14；累加 int32 |
| ② 展开 | 为配合池化，外循环改成“池化位置 (pr,pc) × 2×2 子位置”，共 169×4 次发射 |
| ③ 存储 | 权重 36 字节用 `$readmemh` 载入，综合后是常数；一个池化位置要 4×4 像素，读 4 行存入 `patch[0..3]` |
| ④ 时序 | step0–4 取 4 行；step5–8 每拍发射一个子位置；流水级 1 是乘法，级 2 是 9 项求和加偏置；每个位置 11 拍 |
| ⑤ 接口 | 内部模块，用 `v1/v2` 标记数据有效，`first/last` 标记 2×2 的首尾 |

```verilog
// 流水 1：36 个乘法，注意 uint8 先补 0
prod[c*9+k] <= $signed({1'b0, px[k]}) * $signed(conv_w[c*9+k]);
// 流水 2：求和
acc[c] <= conv_b[c] + prod[c*9+0] + ... + prod[c*9+8];
```

**易错点：** PyTorch 的“卷积”其实是互相关，权重不翻转；权重索引是 `c*9 + kr*3 + kc`。对不上时，先拿 `conv_acc_int32` 逐个比。

## 7. 模块 3：ReLU/重量化（np.clip → 比较器 + 移位 + 饱和）

这一层在硬件里最便宜：没有乘法，只是纯组合逻辑，直接跟在卷积流水后面。

**软件：**

```python
positive = np.maximum(conv_acc, 0)
rounded  = (positive + 128) >> 8          # F=14 → F=6
relu_u8  = np.clip(rounded, 0, 255)
```

**硬件：** 每个 Python 操作正好对应一个电路元件。

| Python | 硬件电路 |
| --- | --- |
| `max(x, 0)` | 看符号位 `a[31]`，为 1 就输出 0 |
| `+128` | 一个加法器 |
| `>> 8` | 不耗资源，只是取 `[15:8]` 这几根线 |
| `clip(0,255)` | 比较器 + 多路选择 |

```verilog
function [7:0] relu_req(input signed [31:0] a);
  reg [32:0] t;
  begin
    if (a[31]) relu_req = 0;
    else begin
      t = ({1'b0,a} + 128) >> 8;
      relu_req = (t > 255) ? 255 : t[7:0];
    end
  end
endfunction
```

**易错点：** 饱和必须在截取低 8 位之前做，否则 256 会变成 0。`接口说明.md` 里特别强调了这一点。

## 8. 模块 4：池化（reshape.max → 流水中边算边取最大）

硬件关键技巧是“融合”：不存 26×26 的 ReLU 结果，卷积按 2×2 的顺序产生输出，每来一个就和当前最大值比。这样省掉 2,704 字节存储。

**软件：** `pool = relu.reshape(4, 13, 2, 13, 2).max(axis=(2, 4))`

**改写成硬件思路：**

```python
for pr in range(13):
  for pc in range(13):
    for sp in range(4):                   # 2×2 子位置，每拍一个
      r = relu(conv(2*pr + sp//2, 2*pc + sp%2))
      mx = r if sp == 0 else max(mx, r)
    pool[c][pr][pc] = mx                  # sp==3 时写出
```

| 步骤 | 这一层的做法 |
| --- | --- |
| ① 定点化 | uint8 无符号比较，尺度不变 |
| ② 展开 | 4 个通道并行，各有一个 `mx[c]` 寄存器 |
| ③ 存储 | 输出按通道分 4 个库 `pool_b0..3[169]`，这样同一拍能写 4 个地址 |
| ④ 时序 | `first` 拍直接覆盖 `mx`，`last` 拍写入池化缓存 |
| ⑤ 接口 | 池化缓存同时也是调试点，仿真里直接拿它和 `pool_uint8` 比 |

**为什么分 4 个库：** 一块 RAM 每拍只能写 1 个地址。这是软件里没有的限制，是硬件设计中最常见的“存储带宽”问题。

## 9. 模块 5：展平 + 全连接（矩阵乘 → 10 路并行 MAC + 打包 ROM）

展开方式：每拍读 1 个激活值，同时乘 10 个类别的权重，676 拍算完。展平在硬件里不占任何资源，只是一种地址计算规则。

**软件：**

```python
flat   = pool.reshape(676)                  # idx = c*169 + r*13 + col
scores = flat @ fc_w.T + fc_b               # [10] int32
```

**改写成循环：**

```python
acc = fc_b.copy()
for idx in range(676):                      # ← 计数器轮流
  x = flat[idx]
  for k in range(10):                       # ← 并行：10 个 MAC
    acc[k] += x * fc_w[k][idx]
```

| 步骤 | 这一层的做法 |
| --- | --- |
| ① 定点化 | uint8 × int8，累加 int32，F=12；实测范围 ±12 万，远小于 int32 上限 |
| ② 展开 | 内循环 10 类并行，外循环 676 次用计数器 |
| ③ 存储 | 关键：权重按“每行一个 idx、一行装 10 类”重排，成为 676 行 × 80 位的 ROM，一次读就拿到 10 个权重。这就是 `gen_mem.py` 生成 `fc_weight_packed.mem` 的原因 |
| ④ 时序 | BRAM 是同步读，地址发出后下一拍才有数据，所以用 `fc_v` 延迟 1 拍再累加 |
| ⑤ 接口 | 开始时用偏置初始化 10 个累加器 |

```python
# gen_mem.py 里的软→硬数据重排
for i in range(676):
    line = "".join(hex8(fw[k, i]) for k in range(9, -1, -1))   # 类 0 在低字节
```

**这是软硬转换中最重要的一条经验：** 数据怎么存，要按硬件怎么读来决定，不要按软件原来的形状存。

## 10. 模块 6：argmax（np.argmax → 顺序比较器）

用 10 拍顺序比较，每拍只有一个 32 位比较器。这样时序最宽松，多出的 10 拍相对 2800 拍可以忽略。

| Python | 硬件 |
| --- | --- |
| `scores.argmax()` | `for i in 1..9: if score[i] > best: best = i` |
| 并列时取第一个 | 用严格大于 `>`，不要用 `>=` |
| int32 有符号 | 累加器必须声明成 `reg signed [31:0]`，否则负分数会被当成很大的正数 |

算完后把 10 个分数锁存到 `score_o`。这样下一帧计算时，PS 读到的寄存器仍是完整的上一帧结果。

## 11. 模块 7：接口层（函数调用 → AXI-Lite + DMA + PYNQ 驱动）

软件里的 `model.forward(img)` 在板上被拆成三件事：写数据、等完成、读结果。每件事都有对应的硬件通道。

| 软件动作 | 硬件通道 | PYNQ 代码 |
| --- | --- | --- |
| 传入 784 字节 | DMA MM2S → `s_axis` | `dma.sendchannel.transfer(in_buf)` |
| 拿回分数 | `m_axis` → DMA S2MM（44 字节） | `dma.recvchannel.transfer(out_buf)` |
| 读状态 / 周期 / 分数 | AXI-Lite 寄存器 | `ip.read(0x04)` |
| 复位 | CTRL.bit0 | `ip.write(0x00, 1)` |

调用时序：

1. PS：`recv.transfer(out_buf)`，先挂好接收
2. PS：`send.transfer(in_buf)`
3. DMA → CNN：196 拍 + TLAST
4. CNN：约 2550 拍计算
5. CNN → DMA：11 个字 + TLAST
6. DMA → PS：`wait()` 返回

顺序很重要：先挂接收再发送，否则输出流没人接，核会卡在 S_OUT。

要注意的坑：

- **缓存一致性**：必须用 `pynq.allocate` 分配缓冲区，不能直接传 numpy 数组。
- **.bit 和 .hwh 要同名同版本**：否则 `Overlay` 找不到 IP 名，或者地址不对。
- **先读 ID 寄存器**：读到 `0x434E4E01` 说明 bit 文件和地址都对，这是上板第一步。
- **DMA 长度寄存器位宽** 要设为 14 位以上。

## 12. 验证体系：三层对齐

每一层验证的标准都一样：和 `hardware_reference.npz` 逐位一致，不接受“差不多”。一旦不一致，就从前往后找第一个对不上的层。

| 层次 | 工具 | 比什么 | 状态 |
| --- | --- | --- | --- |
| ① 软件自测 | `04_验证/验证模型.py` | 参数哈希、十张图逐层结果 | 已有 |
| ② RTL 仿真 | `tb_cnn_accel.v` + iverilog / xsim | 池化 676 字节、十类分数、argmax、TLAST 错误帧、反压 | 已通过 |
| ③ 板上自测 | `CnnFpga.self_test()` | 同样十张图经 DMA 跑一遍，比分数并记录延迟 | 待上板 |

定位问题的顺序：`conv_acc_int32 → relu_uint8 → pool_uint8 → scores_int32`。

验证完十张图后还要补的边界用例：全 0 图、全 255 图、随机图、连续 1000 帧压力测试。生成方法：在 Python 里造图 → 用 `integer_inference.py` 算出标准答案 → 写成 `.mem` 给 testbench。

## 13. 开发顺序与里程碑

RTL 和仿真已完成（M2–M3），接下来的重点是 Vivado 和上板。每个里程碑都有明确的“完成标准”，没达到就不进入下一步。

| 里程碑 | 内容 | 完成标准 | 状态 |
| --- | --- | --- | --- |
| M1 | 把 Python 改写成显式循环版，对照原版验证 | 循环版输出和原版一致 | 可选（学习用） |
| M2 | 写 cnn_core：输入 → 卷积 → 池化 → FC → argmax | 仿真十张图逐层 PASS | 完成 |
| M3 | 加 AXI-Lite、顶层、宏开关 | 三种宏组合仿真 PASS | 完成 |
| M4 | Vivado 工程：加源文件，综合 cnn_accel_top | 无 error，100 MHz 时序满足（WNS ≥ 0） | 待做 |
| M5 | Block Design：PS + DMA + CNN，生成 bit/hwh | Validate Design 通过，导出 bit + hwh | 待做 |
| M6 | 上板：读 ID → self_test | 10/10 一致，记录延迟 | 待做 |
| M7 | 接摄像头预处理，端到端演示 | 实拍数字识别正确 | 待做 |
| M8 | 优化：卷积和 FC 流水并行、FC 4 通道并行 | 周期数降到 2000 拍以下，结果仍一致 | 加分项 |

每次改 RTL 后都先跑一遍 M2 的仿真再综合。仿真几秒钟就能跑完，综合加生成 bit 要 10–20 分钟。
