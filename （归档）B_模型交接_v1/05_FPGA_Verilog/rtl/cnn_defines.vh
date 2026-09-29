// =============================================================================
// cnn_defines.vh —— DigitCNN int8 加速器：所有“端口 / 接口 / 地址”相关的宏集中在这里
//
// 目标板：Zynq-7015 (xc7z015clg485-?)，PYNQ 2.7 镜像
// 结构  ：uint8[28,28] → Conv(1→4,3×3) → ReLU/>>8/饱和 → MaxPool2×2 → FC(676→10)
//
// ★ 标 [待确认] 的宏需要和 Vivado Block Design / PYNQ 驱动一起确定后再改。
//   改完宏后：RTL、testbench 自动跟随；pynq/cnn_fpga_driver.py 顶部同名常量也要同步改。
// =============================================================================
`ifndef CNN_DEFINES_VH
`define CNN_DEFINES_VH

// ---------------------------------------------------------------------------
// 1. 端口开关（注释掉即不生成对应端口）
// ---------------------------------------------------------------------------
`define CNN_USE_AXI_LITE          // [待确认] AXI4-Lite 从口：状态/结果/周期计数/软复位（PYNQ MMIO 读写）
`define CNN_USE_M_AXIS            // [待确认] AXI4-Stream 输出十类分数（接 AXI DMA S2MM）；关掉则只能用 AXI-Lite 读分数
`define CNN_USE_IRQ               // [待确认] 完成中断输出（接 PS IRQ_F2P，经 AXI Interrupt Controller）
`define CNN_OUT_ARGMAX_WORD       // [待确认] 输出流在 10 个分数后再追加 1 个字 = argmax 类别（共 44 字节）

// ---------------------------------------------------------------------------
// 2. AXI4-Stream 输入（接 AXI DMA MM2S）
//    每拍装 CNN_S_AXIS_DATA_W/8 个像素，像素序号小的放低字节（与内存小端一致）
//    必须满足 28 % (位宽/8) == 0，因此只允许 8 / 16 / 32
// ---------------------------------------------------------------------------
`define CNN_S_AXIS_DATA_W   32    // [待确认] 与 DMA "Stream Data Width" 一致
`define CNN_IN_PIXELS       784   // 一帧 28×28 = 784 字节，最后一拍必须 TLAST=1

// ---------------------------------------------------------------------------
// 3. AXI4-Stream 输出（接 AXI DMA S2MM），固定 32 bit：每拍一个 int32 分数
// ---------------------------------------------------------------------------
`define CNN_M_AXIS_DATA_W   32    // 不要改：输出按 int32 打包
`ifdef CNN_OUT_ARGMAX_WORD
  `define CNN_OUT_WORDS     11    // 10 个分数 + 1 个 argmax
`else
  `define CNN_OUT_WORDS     10
`endif

// ---------------------------------------------------------------------------
// 4. AXI4-Lite 从口
// ---------------------------------------------------------------------------
`define CNN_AXIL_ADDR_W     8     // [待确认] 地址位宽（256 字节寄存器空间，BD 中分配 4K/64K 均可）
`define CNN_AXIL_DATA_W     32

// 寄存器偏移（字节地址）[待确认]
`define CNN_REG_CTRL        8'h00 // W : bit0 软复位(自清零)  bit1 中断使能     R: bit1 中断使能
`define CNN_REG_STATUS      8'h04 // R : bit0 busy  bit1 done(粘滞)  bit2 tlast_err(粘滞)  bit3 空闲等待输入
                                  // W1C: 写 1 清 bit1 / bit2
`define CNN_REG_RESULT      8'h08 // R : [3:0] 预测数字 argmax
`define CNN_REG_CYCLES      8'h0C // R : 上一帧从第一拍输入到结果就绪的时钟周期数
`define CNN_REG_FRAMES      8'h10 // R : 已完成帧数
`define CNN_REG_ID          8'h14 // R : 固定 ID，用于确认 bitstream 版本
`define CNN_REG_SCORE0      8'h40 // R : 0x40,0x44,...,0x64 → 类别 0..9 的 int32 分数

`define CNN_IP_ID           32'h434E_4E01  // "CNN" + 版本 01

// ---------------------------------------------------------------------------
// 5. 参数 ROM 初始化文件（Vivado 中把 mem/ 下四个文件加入工程；仿真时放在运行目录或写全路径）
// ---------------------------------------------------------------------------
`define CNN_CONV_W_FILE     "conv_weight_int8.mem"
`define CNN_CONV_B_FILE     "conv_bias_int32.mem"
`define CNN_FC_W_FILE       "fc_weight_packed.mem"
`define CNN_FC_B_FILE       "fc_bias_int32.mem"

// ---------------------------------------------------------------------------
// 6. 模型固定尺寸（改动即新模型版本，一般不动）
// ---------------------------------------------------------------------------
`define CNN_IMG_W           28
`define CNN_CONV_CH         4
`define CNN_POOL_W          13
`define CNN_FC_IN           676
`define CNN_FC_OUT          10
`define CNN_REQ_SHIFT       8     // conv 累加 F=14 → 激活 F=6，右移 8，+128 四舍五入

`endif
