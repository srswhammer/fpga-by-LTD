# 复现与来源

## 常规交接复现

仅需 NumPy/Pillow，执行根目录 `verify_release.py` 和 `demo.py`，无需下载 MNIST。最初电脑环境为 Python 3.12，NumPy 2.5.3、Pillow 12.3.0；打包没有附带 .venv、系统依赖或个人 Jupyter 配置。板上兼容性尚未验证。

## 重跑模型实验（B）

`reproduction/` 保留原始脚本、checkpoint、数据划分和候选记录；模型文件原字节保持不变，以通过 checkpoint 内的模型哈希检查。`requirements.lock.txt` 是原电脑环境记录，不是板上安装清单。

在独立电脑环境安装 PyTorch、torchvision、NumPy、Pillow、matplotlib，然后从 reproduction 目录执行：

```python
from torchvision.datasets import MNIST
MNIST("data", train=True, download=True)
MNIST("data", train=False, download=True)
```

```text
python evaluate_baseline.py
python validate_software_prediction.py
python verify_float_reference.py
python design_fixed_point_candidate.py
python verify_fixed_point_candidate.py
python evaluate_integer_v1.py
python verify_integer_v1.py
```

全新训练使用 `python train_baseline.py --run-dir runs/retrain_v1`，避免覆盖随包 checkpoint。训练参数：55k/5k 固定划分 seed=42、Adam、lr=0.001、batch=64、5 epochs，按验证集选最佳 checkpoint。CPU 库版本或平台变化可能产生浮点细微差异；整数结果应完全一致。

## 来源与 AI 协作

- 数据集：MNIST，通过 torchvision.datasets.MNIST 下载。包中包含少量验证样本及由其生成的示例，未包含完整训练集。
- 训练依赖 PyTorch/torchvision，整数参考依赖 NumPy，图片处理依赖 Pillow，Notebook 依赖 Jupyter。它们是外部开源依赖，许可证以安装包所附内容为准。
- 本模型的训练、量化、接口说明与交接脚本由 B 在 AI 辅助下逐步生成、执行和检查；不是人工逐行独立编写的声明。
- 未从某个整套 CNN FPGA 仓库直接移植 RTL；此包没有 RTL/HLS、bit/hwh 或已验证的 DMA 驱动。
- 本包记录本地实际脚本、数值结果和验证边界，不宣称竞赛最终提交材料、开源许可审查或硬件验收已完成。

后续将 A 的实现、C 的实拍记录和团队复现结果补入仓库，并按竞赛实际规则整理最终材料。
