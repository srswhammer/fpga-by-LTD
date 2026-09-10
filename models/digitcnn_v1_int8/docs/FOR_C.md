# 给 C：先跑软件入口，再替换成 Overlay

## 第一次运行

从交接目录按 README 安装 NumPy/Pillow，然后运行 `python demo.py` 或打开 `demo.ipynb`。此入口不需要 PyTorch、checkpoint 或 MNIST 下载。

```python
from preprocess import preprocess_digit_image
from integer_inference import IntegerDigitCNN

model = IntegerDigitCNN.from_directory()  # 只加载一次
prepared = preprocess_digit_image("examples/camera_like_00_label_1.png")
pixels = prepared.normalized_uint8        # [28,28], uint8
result = model.forward(pixels)
print(int(result.predictions[0]))
print(result.scores_int32[0])             # 十个 int32，按数字 0～9
```

返回的是 CPU 整数参考结果。以后保留同样的 `pixels`，将 `model.forward(pixels)` 替换成已验证的 DMA/Overlay 调用即可。没有提供假定 IP 名称或寄存器地址的硬件驱动。

## 摄像头输入要求

先截取一个完整数字的 ROI，背景单一、数字不贴边。支持文件、PIL 图片、灰度数组或 RGB/RGBA 数组。OpenCV 的 BGR 帧应先换成 RGB；不要把 BGR 当 RGB，也不要重复预处理已经标准化的输入。

预处理依次灰度化、估计边缘背景、统一白字黑底、裁剪数字、保持比例缩放并居中；空白/低对比度会抛出 ValueError。界面应显示“未找到清晰数字”，而不是复用上一帧预测。它不能检测多数字或保证拒绝所有非数字物体。

当前代码将满足黑边条件的 28×28 白字黑底图视为已经标准化，保留原字节。这是 v1 的行为约定；对任意手绘小图，建议提供较大 ROI。

## 与 A 确认后再写的部分

输入缓冲区类型与字节顺序、DMA 的输入输出长度、启动先后、done 和异常状态、缓存 flush/invalidate、超时处理。草案是每图 784 字节输入、40 字节 int32 分数输出，但不能当成已有硬件配置。

板上 NumPy/Pillow 版本需实测；requirements 是 API 范围，不等于所有组合已验证。若板上环境较旧，先报告 Python 和库版本，保留原系统镜像。

## 实拍验收

先每类收集约 10 张，即总计至少 100 张有标签的真实 ROI，记录原图、28×28 图、标签、预测与光照情况。当前 94.10% 是合成拍照图结果，不是 OV5640 实拍成绩。不要上传含个人信息的无关整帧。
