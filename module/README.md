# module：手写数字识别 CNN 加速器（Verilog）

逐层对齐 `B_模型交接_v1/02_模型核心/integer_inference.py` 与
`B_模型交接_v1/03_FPGA交接/接口说明.md` 数值合同的 RTL 实现，
已用固定十张图片仿真验证十个 int32 分数和预测数字与软件标准答案
逐比特一致。

```text
module/
├── rtl/
│   ├── cnn_top.v            顶层：装图片 -> start/busy/done -> 十个分数+预测数字
│   ├── input_buffer.v       784 字节输入图片缓存
│   ├── conv_pool_unit.v     卷积 + ReLU/右移量化 + 2x2 最大池化
│   ├── flatten_buffer.v     676 字节展平结果缓存
│   └── fc_argmax_unit.v     全连接 + argmax（并列取较小编号）
├── params/                  参数 .mem（不入库，运行下面脚本现场生成）
├── scripts/
│   ├── 同步参数.py           把 generated_fpga/*.mem 复制成 ASCII 路径副本
│   └── 生成仿真数据.py       从 hardware_reference.npz 导出十张图片的仿真向量
└── sim/
    ├── tb_cnn_top.v         十张固定图片自检 testbench
    └── tb_*.mem             仿真向量（由 生成仿真数据.py 生成）
```

## 结构与数值规则

```text
uint8[28,28] -> Conv(1->4,3x3,stride1,pad0) -> ReLU/右移8位/饱和[0,255]
             -> MaxPool(2x2,stride2) -> Flatten(676) -> FC(676->10) -> argmax
```

- 卷积：互相关（不翻转核），int32 累加：`bias_int32 + sum(input_uint8*weight_int8)`
- 量化：`sat_u8((max(conv_acc,0) + 128) >> 8)`，右移位数 = 14-6 = 8（来自
  `config.json` 里卷积累加器和 ReLU 激活的小数位之差）
- 池化：每通道 2x2、stride=2 无符号最大值
- 展平地址：`channel*169 + row*13 + column`
- 全连接：`bias_int32 + sum(flatten_uint8*weight_int8)`，不做额外右移
- 并列最大分数时取较小类别编号（按 0..9 顺序比较，只用严格 `>` 更新最优解）

以上每一条都能在 `integer_inference.py` 里找到对应实现，RTL 修改前先看那份代码，
不要凭記憶改右移位数或展平顺序。

## 运行仿真

首次运行，或模型/参数更新后，先重新生成参数文件：

```bash
python "B_模型交接_v1/03_FPGA交接/导出FPGA参数.py"   # 生成 B_模型交接_v1/generated_fpga/*.mem
python "module/scripts/同步参数.py"                    # 复制成 module/params/ 下的 ASCII 路径副本
python "module/scripts/生成仿真数据.py"                # 生成 module/sim/tb_*.mem 测试向量
```

（`module/params/` 用 ASCII 路径是因为部分工具的 `$readmemh` 处理含中文的
相对路径会读取失败——曾用 Icarus Verilog 实测复现过，改成 ASCII 路径后正常。
仿真/综合工具选型确定后可以再评估要不要恢复引用 `generated_fpga/`。）

用 Icarus Verilog 跑十张图片自检：

```bash
cd module/sim
iverilog -o sim.vvp -g2005 ../rtl/input_buffer.v ../rtl/flatten_buffer.v \
    ../rtl/conv_pool_unit.v ../rtl/fc_argmax_unit.v ../rtl/cnn_top.v tb_cnn_top.v
vvp sim.vvp
```

预期输出十行 `[PASS] image=N digit=N (scores all matched)` 加
`==== ALL 10 IMAGES PASSED ====`。若换用 ModelSim/Vivado 仿真器，把这几个
文件加入工程、`tb_cnn_top` 设为顶层即可，激励和参数加载方式不变。

## 接口

`cnn_top` 目前是最简单的寄存器/存储器式握手，还没有绑定 AXI-Lite/AXI-Stream
——`接口说明.md` 里这部分协议（AXI 数据宽度、DMA TLAST/TKEEP、寄存器地址等）
还没和 C 确认。等协议定下来，在 `cnn_top` 外面包一层 AXI 适配壳即可，内部数据通路不用改。

```verilog
cnn_top u_cnn (
    .clk(clk), .rst_n(rst_n),
    .img_wr_en(...), .img_wr_addr(...), .img_wr_data(...), // 逐字节写 784 个像素
    .start(...), .busy(...), .done(...),                    // 图片装完后拉高一拍 start
    .scores_packed(...),  // scores_packed[32*i +: 32] 是第 i 类 int32 分数
    .digit(...)           // 预测数字 0..9
);
```

## 已知限制 / 后续可做的事

- **时序**：卷积用逐抽头串行 MAC（每个池化窗口约 41 拍，全图约 2.8 万拍），
  全连接约 6800 拍，单张图片约 3.5 万个时钟周期。先保正确性，之后要提吞吐量
  可以把 3x3 九个抽头或 2x2 四个子位置改成并行 MAC。
- **存储**：`input_buffer`/`flatten_buffer` 用异步读的 `reg` 数组实现（分布式
  RAM 风格），没有用 BRAM 原语；这两个缓存都很小（784B/676B），综合工具通常会
  自动映射成合适的资源，一般不需要手工干预。
- **验证范围**：目前只对齐了 `hardware_reference.npz` 的固定十张图片（对应
  `接口说明.md` 第 5 步"第 0 张完全一致后扩展到十张"）；全零、全 255、随机和
  误分类输入还没补。
- 还没上板测过资源占用、时序收敛和实际延迟/吞吐——这些要等综合和板级验证。
