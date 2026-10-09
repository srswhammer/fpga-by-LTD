# v2.4 数独模型交接包

识别空白和数字1～9，使用INT8参数。板上前处理流程：

**已切分的单格图片 → preprocess.py → 28×28 UINT8 → board_bram.py → BRAM → FPGA CNN**

## 板上使用

```python
from board_bram import open_bram, write_image

bram = open_bram(overlay, 'cnn_bram')  # 填实际AXI BRAM Controller名称
pixels = write_image(bram, cell_image, polarity='dark_on_light')
```

复用已经加载的PYNQ Overlay。写入完成后，由外部控制程序启动CNN，并等识别结束后再写下一格。
这里只做预处理和BRAM写入，不做硬件预测控制，不自动加载bit，不依赖Torch。

数据格式、地址和硬件配合要求见 [板上BRAM接口.md](板上BRAM接口.md)。

## 文件用途

| 文件 | 用途 |
|---|---|
| `preprocess.py` | 单格灰度化、裁剪、缩放居中；另保留离线PNG/HEX导出功能 |
| `board_bram.py` | PYNQ映射BRAM、写入784字节像素 |
| `predict.py` | 电脑上的CNN软件参考计算、预测和自检；板上写图不调用它 |
| `weights/int8.npz`、`int8.json` | INT8权重、INT32偏置，以及布局和量化规则 |
| `weights/fp32.pt` | 浮点参考权重，板上INT8流程不用 |
| `权重TXT/` | 两层卷积和全连接层的整数权重、偏置数组 |
| `预处理图像(示例)/` | 270张标准输入PNG、HEX及软件参考结果，供仿真对照 |
| `self_test.npz` | 软件运行自检样本 |
| `run.ps1`、`requirements.txt` | 电脑端启动和依赖；板上写图无需使用 |
| `板上BRAM接口.md` | 简洁的板上调用与输入协议 |

板上依赖PYNQ、NumPy、Pillow。输入为单格，不是整张数独；本包不负责棋盘定位、切分、求解及显示。

## 模型与结果

结构：`1×28×28 → Conv3×3(4)+ReLU → Pool2×2 → Conv3×3(8)+ReLU → Pool2×2 → Flatten200 → FC10`。
卷积步长1、无padding；池化步长2、向下取整，共2346个参数。
输入像素保留UINT8；INT8运算和各层移位规则见 `predict.py`、`weights/int8.json`。

输出9位独热：空白为000000000；数字n对应 `1 << (n-1)`，即1为000000001，9为100000000。

已有软件验证：标准测试INT8为9878/10020（98.58%）；自采270张为269/270（99.63%），有一张4被识别成9。
照片结果不是独立测试集或整机准确率；当前权重、预处理算法不因BRAM传输而改变。尚未做真实板端验证。

## 电脑软件参考（可选）

```powershell
python -B predict.py --self-test
python -B predict.py --image "单格.png" --backend int8 --polarity dark_on_light
```

电脑参考计算需要安装 `requirements.txt` 中的依赖；板上写图不要使用这份Torch依赖清单。
标准28×28白字黑底输入用light_on_dark或auto，避免再次反色。
