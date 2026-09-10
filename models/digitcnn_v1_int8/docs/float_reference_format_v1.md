# float32 逐层参考数据包 v1

状态：第五步已完成。该数据包是 `digitcnn_v1_fp32_baseline` 的软件标准答案，不是最终 PL 量化格式。

## 为什么不能只给 A 一个最终数字

如果软件预测是 7，硬件预测是 2，只比较最终结果无法知道错误来源。逐层参考把网络拆成五个检查点：卷积、ReLU、池化、展平和全连接分数。A 可以从第一层开始比较，第一处不一致的位置就是重点排查位置。

参考包位于 `../reproduction/runs/baseline_v1/float_reference_v1/`。`manifest.json` 记录版本、数组形状、数据类型、布局和 SHA-256；任何文件被意外替换后，验证程序都会发现。

## 固定样本

从训练阶段保留的 5000 张验证集中，为数字 0～9 分别选取第一张能被当前模型正确识别的图片。固定顺序为：

```text
sample_id:  0 1 2 3 4 5 6 7 8 9
true_label: 0 1 2 3 4 5 6 7 8 9
prediction: 0 1 2 3 4 5 6 7 8 9
```

没有使用官方 MNIST 测试集选择这些样本。`samples.csv` 保存样本编号、标签、原 MNIST 索引和验证集位置。

## 主要文件及形状

| 文件 | dtype | 形状 | 含义 |
|---|---|---|---|
| `inputs/fixed_inputs_uint8.npy` | uint8 | `[10,28,28]` | 白字黑底，0～255，给软件边界和后续搬运测试 |
| `inputs/fixed_inputs_float32.npy` | float32 | `[10,1,28,28]` | uint8 除以 255 后的 NCHW 模型输入 |
| `weights/conv_weight_float32.npy` | float32 | `[4,1,3,3]` | 卷积权重 |
| `weights/conv_bias_float32.npy` | float32 | `[4]` | 卷积偏置 |
| `weights/fc_weight_float32.npy` | float32 | `[10,676]` | 全连接权重 |
| `weights/fc_bias_float32.npy` | float32 | `[10]` | 全连接偏置 |
| `activations/conv_output_float32.npy` | float32 | `[10,4,26,26]` | 卷积层输出，尚未 ReLU |
| `activations/relu_output_float32.npy` | float32 | `[10,4,26,26]` | 负值置零后的输出 |
| `activations/pool_output_float32.npy` | float32 | `[10,4,13,13]` | 2×2 最大池化输出 |
| `activations/flatten_output_float32.npy` | float32 | `[10,676]` | 展平结果 |
| `activations/scores_float32.npy` | float32 | `[10,10]` | 0～9 的十个原始分数 |
| `activations/predictions_int64.npy` | int64 | `[10]` | 每行最大分数的索引 |

`reference_bundle_v1.npz` 是以上数组的压缩副本，便于一次加载。`.npy` 文件带有 NumPy 文件头，不能把整个文件直接当成无头二进制像素流发送给 FPGA。

## 布局规则

所有数组都是 C/行优先排列，最后一维变化最快。

卷积权重：

```text
conv_weight[输出通道, 输入通道, 卷积核行, 卷积核列]
```

全连接权重：

```text
fc_weight[输出数字类别, flatten_index]
```

池化结果的展平规则：

```text
flatten_index = channel × 13 × 13 + row × 13 + column
```

所以同一个通道内先连续保存一行，再保存下一行；一个通道的 169 个数保存完后才进入下一通道。

## 每层计算原理

卷积层没有 padding，stride 为 1。某个输出位置的计算为：

```text
conv[n,oc,r,c]
= conv_bias[oc]
  + Σ input[n,ic,r+kr,c+kc] × conv_weight[oc,ic,kr,kc]
```

当前输入通道数为 1、卷积核为 3×3，因此每个输出值包含 9 次权重乘法和一次偏置。

ReLU：`relu = max(conv, 0)`。

最大池化：每个通道把不重叠的 2×2 区域取最大值，将 26×26 变成 13×13。

全连接：

```text
score[digit] = fc_bias[digit]
             + Σ flatten[index] × fc_weight[digit,index]
```

最终取十个 `score` 的最大值索引，不需要在 PL 中计算 softmax。

## 实际 3×3 卷积手算案例

手算文件使用数字 0（sample 0）、输出通道 1、输出位置 `(row=5,column=13)`：

- 卷积偏置：`0.3280968070`
- 九次乘法按卷积核行列依次累加；每一项见 `manual_check/manual_conv_terms.csv`
- float32 顺序累加结果：`2.3622555733`
- PyTorch 结果：`2.3622555733`
- 本机差值：`0`

这个案例证明了输入窗口位置、卷积核方向和权重布局。PyTorch `Conv2d` 使用互相关规则，不会把 3×3 卷积核上下左右翻转。

## 如何验证

在项目 PowerShell 中运行：

```powershell
cd reproduction
python verify_float_reference.py
```

验证程序会检查：文件哈希、数组形状和类型、uint8 转 float32、权重与 checkpoint、展平顺序、压缩包副本，以及从导出权重重新计算的每一层输出。

同一环境的软件复算要求逐位相同。未来量化 PL 输出不应直接与这些 float32 数值逐位比较；第 6 步确定舍入、饱和、缩放和累加规则后，第 7 步会生成整数参考数据。
