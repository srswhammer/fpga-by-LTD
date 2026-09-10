# 给 A：先验证数值，再连接总线

## 这次接收的内容

1. [发布配置](../release/config.json)：软件数值规则已稳定。
2. [参数文件](../release/parameters/)：四组 MEM 可用 `$readmemh` 初始化仿真存储器；BIN 布局见 [binary_layout.json](../release/parameters/binary_layout.json)。
3. [固定输入与逐层结果](../release/reference/)：10 个样本，每个编号就是对应的数字。选样时要求当前模型识别正确；这 10 张只用于联调，不代表精度评估。
4. [数值推导记录](fixed_point_candidate_v1.md)：历史候选说明。该格式已在第七步通过软件精度验收，以当前发布 config 为准。
5. [AXI 草案](hardware_interface_draft_v1.md)：可讨论的总线方案，尚未在板上验证。

## 必须严格一致的数值规则

`F` 的含义为 `实际值=整数×2^-F`。像素虽然与预处理输出是同一字节，但整数模型将其解释为 p/256；不要提前再除一次。

| 数据 | 类型 | F |
|---|---|---:|
| 输入 | uint8 | 8 |
| 卷积权重 | int8（补码） | 6 |
| 卷积偏置/累加 | int32 | 14 |
| ReLU 后/池化 | uint8 | 6 |
| FC 权重 | int8（补码） | 6 |
| FC 偏置/分数 | int32 | 12 |

卷积为互相关，不翻转卷积核；stride=1，padding=0。卷积后执行 `min(255,(max(acc,0)+128)>>8)`。先用足够位宽完成加法和比较，再截成 uint8，不能先截低八位。乘法要正确扩展 uint8 输入与 int8 负权重，禁止将输入的 255 误当成 -1。

池化是每通道 2×2、stride=2 的无符号最大值。展平索引 `channel*169+row*13+column`。所有 FC 分数同尺度，按有符号 int32 比较；遇到相等最大值取较小的数字编号，与 NumPy argmax 一致。

## 第一周联调顺序

1. 在电脑跑 `verify_release.py`，先确认数据包完整。
2. 用 sample 0 写 testbench：先逐值对比 `conv_acc_int32.npy`。
3. 依次比较 ReLU、pool、flatten、scores；任何整数值不一致都应先解决。
4. 扩展到全部 10 张。正确结果为 0～9；逐层全等比仅标签一致更重要。
5. 再接 AXI/DMA，与 C 共同确认寄存器、打包、TLAST、TKEEP、缓存同步和启动顺序。
6. 板上跑同样输入，对比软件答案；保存 Vivado 资源与时序报告、单张延迟和端到端延迟。

目前提供的是 NPY 参考，可通过 `np.load()` 读取后 `.tofile()` 转无头字节；NPY 文件有头，不能整文件直接灌进 FPGA。参数 MEM 是每行一个整数，不是四像素 AXI 打包字。

## 需要 A 回复的决定

RTL/HLS 路线、板上 Python/PYNQ 版本、目标时钟、可用资源、已有 DMA 总线宽度、参数 ROM 或运行时加载、调试特征读取方式。软件数值规则变更需 B 重新导出并评估；总线草案可以与 C 协商调整。
