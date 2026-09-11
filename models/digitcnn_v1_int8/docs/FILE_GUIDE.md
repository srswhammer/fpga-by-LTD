# 模型交接包逐文件说明

对应初始模型提交：`5d3193e6d229d275d4826d302f622babf8a8eb4a`。本次补充说明，保留原有目录与模型参数。

## 组长先看：这些文件分成什么用途

本文件夹保存一个模型的运行文件、评估证据和训练复现材料。`release/` 是当前整数发布数据；`reproduction/` 是 B 保留的实验过程。多种权重格式存的是同一套参数，分别方便 Python、Verilog 和二进制加载使用。

**请完整保留 digitcnn_v1_int8 文件夹及内部相对路径。** 阅读分工只决定先看什么，不需要把文件拆出目录。脚本按相对路径找配置、参数和样本，校验清单也与这些文件配套。

| 人员 | 最短阅读与执行顺序 | 初期不必深入阅读 |
|---|---|---|
| 组长 | README → docs/STATUS → 本说明 | 逐样本数组和实验源码 |
| A：硬件 | docs/FOR_A → release/config → parameters → reference；运行 verify_release.py | reproduction 中训练过程 |
| C：联调 | docs/FOR_C → demo.py 或 demo.ipynb → preprocess.py、integer_inference.py | 浮点 checkpoint 和量化推导 |
| B：模型 | docs/REPRODUCIBILITY → reproduction → release/evaluation | 按维护任务查阅 |

A/C 初次运行不需要训练或下载完整 MNIST。当前包没有 RTL/HLS、bit/hwh 或已验证的 DMA 驱动；软件通过验证不代表 FPGA 实现已完成。

## 格式和同名文件怎么区分

| 格式或名称 | 查看与使用方法 |
|---|---|
| .md | Markdown 文档，GitHub 可直接预览。 |
| .py / .ipynb | Python 脚本 / Jupyter 分单元格笔记本，执行方法见 README。 |
| .json / .csv / .png | 配置或结果记录 / 数值表 / 图片，不是都要执行的程序。 |
| .npy | 一个带形状、类型信息的数组，用 np.load 读取；不能把文件头当像素灌入 FPGA。 |
| .npz | 多个具名数组，np.load 后用 .files 查看字段。 |
| .mem | 每行一个展平数据的十六进制文本，按位宽、补码规则用于 Verilog 初始化。 |
| .bin | 无文件头字节，配合 binary_layout.json 的偏移、长度和端序解析。 |
| .pt | 浮点 PyTorch checkpoint，供 B 复现，不是 FPGA bitstream。 |
| 根目录与 reproduction 同名脚本 | 根目录是 A/C 便携入口；reproduction 保留实验流程，默认读写路径不同。 |
| PACKAGE_MANIFEST / release/manifest | 前者校验整包，后者描述并核验发布数据，都不是模型权重。 |

## 按原目录逐文件查看

包内全部 119 个文件在下面各列一行（包含本说明）。点击文件名可查看或下载。NPY 形状/类型及 NPZ 字段从实际文件读取；形状中的第一维 10 通常对应固定十张输入。


### 交接包根目录

| 文件 | 用途与使用者 |
|---|---|
| [.gitignore](../.gitignore) | 忽略虚拟环境、缓存与下载的数据，属于 Git 维护规则。 |
| [PACKAGE_MANIFEST.json](../PACKAGE_MANIFEST.json) | 整个交接目录的 SHA-256 清单（不含自身），供 verify_package.py 核验完整性。 |
| [README.md](../README.md) | 交接首页：运行命令、精度与角色入口，全员先读。 |
| [demo.ipynb](../demo.ipynb) | Jupyter 分单元格演示：加载模型、预测样本及图片、查看结果，C 使用。 |
| [demo.py](../demo.py) | 命令行预测演示，默认跑固定十张，--image 可指定图片，C 首次运行。 |
| [integer_inference.py](../integer_inference.py) | CPU 整数 CNN 运算实现，A 对照数值、C 调用软件预测。此版本默认读取本包 release。 |
| [packaging_checks.json](../packaging_checks.json) | 上次独立整数验证结果与环境记录，不是 FPGA 验证报告。 |
| [preprocess.py](../preprocess.py) | 单数字图片转 uint8[28,28]，处理灰度、背景、裁剪、缩放、居中；C 接摄像头 ROI 后调用。 |
| [preprocessing_parity.json](../preprocessing_parity.json) | 打包时原版与便携预处理的 22 张图片像素一致性记录。 |
| [requirements-notebook.txt](../requirements-notebook.txt) | 增加 JupyterLab/ipykernel 的依赖清单，C 打开 Notebook 时使用。 |
| [requirements.txt](../requirements.txt) | 运行整数模型及预处理的 NumPy/Pillow 依赖范围，A/C 安装使用。 |
| [verify_package.py](../verify_package.py) | 只需 Python 标准库，检查整包文件缺失或改动，下载后先运行。 |
| [verify_packaging.py](../verify_packaging.py) | 用独立标量循环检查 14 张输入整数运算，验证 BIN/MEM 解码和示例调用；会写 packaging_checks.json。 |
| [verify_release.py](../verify_release.py) | 检查发布参数哈希、MEM/BIN/NPY 一致性、固定输入逐层结果和评估记录；会写 release/verification.json。 |

### docs

| 文件 | 用途与使用者 |
|---|---|
| [FILE_GUIDE.md](../docs/FILE_GUIDE.md) | 本说明：逐文件用途、格式关系与阅读分工，组长及全员查看。 |
| [FOR_A.md](../docs/FOR_A.md) | A 接手步骤：定点规则、参数读取、逐层对比与硬件接口协商项。 |
| [FOR_C.md](../docs/FOR_C.md) | C 接手步骤：软件调用、预处理、摄像头 ROI 与 Overlay 联调。 |
| [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md) | 训练复现方法、数据来源、依赖与 AI 协作说明，B 使用。 |
| [STATUS.md](../docs/STATUS.md) | 完成项、待办、接手人和未验证边界，组长先读。 |
| [fixed_point_candidate_v1.md](../docs/fixed_point_candidate_v1.md) | 定点候选阶段设计推导历史，B 解释数值来源时查阅。 |
| [float_reference_format_v1.md](../docs/float_reference_format_v1.md) | 历史浮点参考格式，与 reproduction 中浮点数据配套阅读。 |
| [hardware_interface_draft_v1.md](../docs/hardware_interface_draft_v1.md) | A/C 商定 AXI 和输入输出的草案，不是已实现的硬件驱动。 |
| [integer_release_v1.md](../docs/integer_release_v1.md) | 整数第一版的软件验收说明；当前数值以 release/config.json 为准。 |
| [model_spec_v1.md](../docs/model_spec_v1.md) | 模型结构和基线设计规格，A/B 理解各层及参数时参考。 |

### examples

| 文件 | 用途与使用者 |
|---|---|
| [camera_like_00_label_1.png](../examples/camera_like_00_label_1.png) | 合成拍照风格示例，真实标签为 1；C 测预处理/预测，不是 OV5640 实拍数据。 |
| [camera_like_01_label_4.png](../examples/camera_like_01_label_4.png) | 合成拍照风格示例，真实标签为 4；C 测预处理/预测，不是 OV5640 实拍数据。 |
| [camera_like_02_label_7.png](../examples/camera_like_02_label_7.png) | 合成拍照风格示例，真实标签为 7；C 测预处理/预测，不是 OV5640 实拍数据。 |
| [camera_like_03_label_4.png](../examples/camera_like_03_label_4.png) | 合成拍照风格示例，真实标签为 4；C 测预处理/预测，不是 OV5640 实拍数据。 |
| [camera_like_04_label_7.png](../examples/camera_like_04_label_7.png) | 合成拍照风格示例，真实标签为 7；C 测预处理/预测，不是 OV5640 实拍数据。 |
| [camera_like_05_label_7.png](../examples/camera_like_05_label_7.png) | 合成拍照风格示例，真实标签为 7；C 测预处理/预测，不是 OV5640 实拍数据。 |
| [camera_like_06_label_7.png](../examples/camera_like_06_label_7.png) | 合成拍照风格示例，真实标签为 7；C 测预处理/预测，不是 OV5640 实拍数据。 |
| [camera_like_07_label_2.png](../examples/camera_like_07_label_2.png) | 合成拍照风格示例，真实标签为 2；C 测预处理/预测，不是 OV5640 实拍数据。 |
| [camera_like_08_label_5.png](../examples/camera_like_08_label_5.png) | 合成拍照风格示例，真实标签为 5；C 测预处理/预测，不是 OV5640 实拍数据。 |
| [camera_like_09_label_2.png](../examples/camera_like_09_label_2.png) | 合成拍照风格示例，真实标签为 2；C 测预处理/预测，不是 OV5640 实拍数据。 |
| [camera_like_10_label_6.png](../examples/camera_like_10_label_6.png) | 合成拍照风格示例，真实标签为 6；C 测预处理/预测，不是 OV5640 实拍数据。 |
| [camera_like_11_label_7.png](../examples/camera_like_11_label_7.png) | 合成拍照风格示例，真实标签为 7；C 测预处理/预测，不是 OV5640 实拍数据。 |

### release

| 文件 | 用途与使用者 |
|---|---|
| [config.json](../release/config.json) | 当前整数模型结构、位宽、缩放和数值规则，A 实现硬件的首要依据。 |
| [manifest.json](../release/manifest.json) | 发布数据的元数据与哈希清单，供 verify_release.py 核验，范围不同于整包清单。 |
| [verification.json](../release/verification.json) | 发布参数、逐层结果及精度验收的上次检查记录，硬件尚未验证。 |

### release/evaluation

| 文件 | 用途与使用者 |
|---|---|
| [accuracy_comparison.png](../release/evaluation/accuracy_comparison.png) | 浮点/整数在三种数据上的精度对比图，组长/B 汇报使用。 |
| [official_test_confusion_int8.csv](../release/evaluation/official_test_confusion_int8.csv) | 官方测试集 10000 张的整数混淆矩阵，B 分析错误分布。 |
| [official_test_confusion_int8.png](../release/evaluation/official_test_confusion_int8.png) | 官方测试集的整数混淆矩阵图，B 查看易混数字。 |
| [official_test_per_class_accuracy.csv](../release/evaluation/official_test_per_class_accuracy.csv) | 官方测试集 10000 张各数字分类准确率，B 查薄弱类别。 |
| [official_test_predictions.npz](../release/evaluation/official_test_predictions.npz) | 官方测试集 10000 张逐样本预测相关数组，B 重算统计及定位分歧。 字段：`labels`、`float_predictions`、`float_scores`、`int8_predictions`、`int8_scores`。 |
| [official_test_quantization_disagreements.png](../release/evaluation/official_test_quantization_disagreements.png) | 量化前后预测有分歧的图片示例，B 分析误差。 |
| [summary.json](../release/evaluation/summary.json) | 验证集、官方测试集、合成拍照图的 float/int8 精度、饱和情况和验收汇总。 |
| [synthetic_camera_like_confusion_int8.csv](../release/evaluation/synthetic_camera_like_confusion_int8.csv) | 合成拍照图 1000 张（不是实拍）的整数混淆矩阵，B 分析错误分布。 |
| [synthetic_camera_like_per_class_accuracy.csv](../release/evaluation/synthetic_camera_like_per_class_accuracy.csv) | 合成拍照图 1000 张（不是实拍）各数字分类准确率，B 查薄弱类别。 |
| [synthetic_camera_like_predictions.npz](../release/evaluation/synthetic_camera_like_predictions.npz) | 合成拍照图 1000 张（不是实拍）逐样本预测相关数组，B 重算统计及定位分歧。 字段：`labels`、`float_predictions`、`float_scores`、`int8_predictions`、`int8_scores`。 |
| [validation_confusion_int8.csv](../release/evaluation/validation_confusion_int8.csv) | 验证集 5000 张的整数混淆矩阵，B 分析错误分布。 |
| [validation_per_class_accuracy.csv](../release/evaluation/validation_per_class_accuracy.csv) | 验证集 5000 张各数字分类准确率，B 查薄弱类别。 |
| [validation_predictions.npz](../release/evaluation/validation_predictions.npz) | 验证集 5000 张逐样本预测相关数组，B 重算统计及定位分歧。 字段：`labels`、`float_predictions`、`float_scores`、`int8_predictions`、`int8_scores`。 |

### release/parameters

| 文件 | 用途与使用者 |
|---|---|
| [binary_layout.json](../release/parameters/binary_layout.json) | BIN 各参数段的名称、偏移、长度和形状，解析 BIN 必须参考。 |
| [conv_bias_int32.mem](../release/parameters/conv_bias_int32.mem) | 卷积层偏置，类型 int32。十六进制补码文本，每行一个展平数据，供 A 仿真初始化。 |
| [conv_bias_int32.npy](../release/parameters/conv_bias_int32.npy) | 卷积层偏置，类型 int32。NumPy 单数组，供软件读取检查。 数组：`[4]`，`int32`。 |
| [conv_weight_int8.mem](../release/parameters/conv_weight_int8.mem) | 卷积层权重，类型 int8。十六进制补码文本，每行一个展平数据，供 A 仿真初始化。 |
| [conv_weight_int8.npy](../release/parameters/conv_weight_int8.npy) | 卷积层权重，类型 int8。NumPy 单数组，供软件读取检查。 数组：`[4, 1, 3, 3]`，`int8`。 |
| [fc_bias_int32.mem](../release/parameters/fc_bias_int32.mem) | 全连接层偏置，类型 int32。十六进制补码文本，每行一个展平数据，供 A 仿真初始化。 |
| [fc_bias_int32.npy](../release/parameters/fc_bias_int32.npy) | 全连接层偏置，类型 int32。NumPy 单数组，供软件读取检查。 数组：`[10]`，`int32`。 |
| [fc_weight_int8.mem](../release/parameters/fc_weight_int8.mem) | 全连接层权重，类型 int8。十六进制补码文本，每行一个展平数据，供 A 仿真初始化。 |
| [fc_weight_int8.npy](../release/parameters/fc_weight_int8.npy) | 全连接层权重，类型 int8。NumPy 单数组，供软件读取检查。 数组：`[10, 676]`，`int8`。 |
| [model_parameters_int.npz](../release/parameters/model_parameters_int.npz) | 四组整数参数的合并存储，当前 IntegerDigitCNN 默认加载它。 字段：`conv_weight_int8`、`conv_bias_int32`、`fc_weight_int8`、`fc_bias_int32`。 |
| [model_parameters_v1.bin](../release/parameters/model_parameters_v1.bin) | 同一套参数的无头字节，共 6852 字节，int32 为小端；供硬件加载方案使用。 |

### release/reference

| 文件 | 用途与使用者 |
|---|---|
| [conv_acc_int32.npy](../release/reference/conv_acc_int32.npy) | 卷积乘加并加偏置的 int32 输出，未 ReLU/缩放，A 第一阶段逐值对比。 数组：`[10, 4, 26, 26]`，`int32`。 |
| [fixed_inputs_uint8.npy](../release/reference/fixed_inputs_uint8.npy) | 固定十张原始 uint8 输入，顺序对应数字 0～9。 数组：`[10, 28, 28]`，`uint8`。 |
| [flatten_uint8.npy](../release/reference/flatten_uint8.npy) | 按通道、行、列展平的 676 维特征，A 检查 FC 输入顺序。 数组：`[10, 676]`，`uint8`。 |
| [labels_int64.npy](../release/reference/labels_int64.npy) | 固定输入的真实数字标签。 数组：`[10]`，`int64`。 |
| [pool_uint8.npy](../release/reference/pool_uint8.npy) | 2×2 最大池化特征，A 检查窗口和通道顺序。 数组：`[10, 4, 13, 13]`，`uint8`。 |
| [predictions_int64.npy](../release/reference/predictions_int64.npy) | 分数 argmax 得到的预测标签，仅标签一致不能替代逐层一致检查。 数组：`[10]`，`int64`。 |
| [relu_uint8.npy](../release/reference/relu_uint8.npy) | ReLU、舍入右移和饱和后的 uint8 特征，A 检查数值转换。 数组：`[10, 4, 26, 26]`，`uint8`。 |
| [scores_int32.npy](../release/reference/scores_int32.npy) | 每图十个有符号 int32 分数，A/C 对比最终数值，不是概率。 数组：`[10, 10]`，`int32`。 |

### reproduction

| 文件 | 用途与使用者 |
|---|---|
| [design_fixed_point_candidate.py](../reproduction/design_fixed_point_candidate.py) | 分析浮点范围、设计定点缩放并生成候选整数参数。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [evaluate_baseline.py](../reproduction/evaluate_baseline.py) | 加载浮点 checkpoint，在 MNIST 测试集评估并输出统计/图表。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [evaluate_integer_v1.py](../reproduction/evaluate_integer_v1.py) | 对比浮点/整数精度，生成 reproduction/runs/digitcnn_v1_int8，不会自动更新本包 release。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [export_float_reference.py](../reproduction/export_float_reference.py) | 导出浮点权重、固定输入、逐层结果和手算卷积示例。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [integer_inference.py](../reproduction/integer_inference.py) | CPU 整数 CNN 运算实现，A 对照数值、C 调用软件预测。 【B 的历史/复现材料；A/C 初期可暂不阅读。】此版本默认读取 reproduction/runs/digitcnn_v1_int8。 |
| [model.py](../reproduction/model.py) | 原始 PyTorch CNN 结构，checkpoint 记录其哈希，复现时保持原字节。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [predict.py](../reproduction/predict.py) | 原始含 PyTorch 的预处理和浮点预测入口，C 使用包根目录便携入口。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [requirements.lock.txt](../reproduction/requirements.lock.txt) | 原训练电脑完整依赖版本记录，供 B 追溯，不要直接用来升级 PYNQ。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [train_baseline.py](../reproduction/train_baseline.py) | 训练基线并保存最佳 checkpoint，B 重训时另设 run-dir。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [validate_software_prediction.py](../reproduction/validate_software_prediction.py) | 验证原始图片预测流程，生成合成示例与检查记录，需要 MNIST 数据。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [verify_fixed_point_candidate.py](../reproduction/verify_fixed_point_candidate.py) | 检查候选整数运算和参考结果一致性。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [verify_float_reference.py](../reproduction/verify_float_reference.py) | 核验历史浮点导出与 checkpoint/逐层结果一致性。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [verify_integer_v1.py](../reproduction/verify_integer_v1.py) | 检查实验生成的 reproduction/runs/digitcnn_v1_int8，不是直接检查本包 release。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |

### reproduction/runs/baseline_v1

| 文件 | 用途与使用者 |
|---|---|
| [best_model.pt](../reproduction/runs/baseline_v1/best_model.pt) | 最佳浮点 PyTorch checkpoint，B 复现使用，不能直接加载到 FPGA。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [config.json](../reproduction/runs/baseline_v1/config.json) | 本次浮点基线训练配置与超参数。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [history.csv](../reproduction/runs/baseline_v1/history.csv) | 逐轮训练/验证指标，B 查看学习过程。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [reload_check.npz](../reproduction/runs/baseline_v1/reload_check.npz) | checkpoint 重新加载对照数组，检查保存前后行为。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 字段：`inputs`、`labels`、`expected_scores`。 |
| [smoke_check.json](../reproduction/runs/baseline_v1/smoke_check.json) | 基线训练的小规模运行检查记录。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [split_indices.npz](../reproduction/runs/baseline_v1/split_indices.npz) | 固定训练/验证划分索引，保证复现使用同一划分。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 字段：`train`、`validation`。 |
| [summary.json](../reproduction/runs/baseline_v1/summary.json) | 浮点基线训练结果摘要。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |

### reproduction/runs/baseline_v1/fixed_point_candidate_v1

| 文件 | 用途与使用者 |
|---|---|
| [accuracy_evaluation.json](../reproduction/runs/baseline_v1/fixed_point_candidate_v1/accuracy_evaluation.json) | 候选精度评估记录，当前验收看 release/evaluation/summary.json。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [candidate_config.json](../reproduction/runs/baseline_v1/fixed_point_candidate_v1/candidate_config.json) | 历史候选定点位宽和缩放配置。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [candidate_parameters_unvalidated.npz](../reproduction/runs/baseline_v1/fixed_point_candidate_v1/candidate_parameters_unvalidated.npz) | 候选参数快照，名字保留当时状态；硬件接手采用 release/parameters。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 字段：`conv_weight_int8`、`conv_bias_int32`、`fc_weight_int8`、`fc_bias_int32`。 |
| [integer_rule_example.json](../reproduction/runs/baseline_v1/fixed_point_candidate_v1/integer_rule_example.json) | 候选整数运算规则的具体示例。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [range_analysis.json](../reproduction/runs/baseline_v1/fixed_point_candidate_v1/range_analysis.json) | 范围分析结果，B 理解缩放与溢出风险。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [verification.json](../reproduction/runs/baseline_v1/fixed_point_candidate_v1/verification.json) | 候选阶段数值一致性检查记录。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |

### reproduction/runs/baseline_v1/float_reference_v1

| 文件 | 用途与使用者 |
|---|---|
| [fixed_inputs_overview.png](../reproduction/runs/baseline_v1/float_reference_v1/fixed_inputs_overview.png) | 历史浮点参考固定十张输入总览。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [manifest.json](../reproduction/runs/baseline_v1/float_reference_v1/manifest.json) | 浮点参考来源、形状、布局、哈希与选样规则。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [reference_bundle_v1.npz](../reproduction/runs/baseline_v1/float_reference_v1/reference_bundle_v1.npz) | 历史浮点参考数组合并存储，便于一次读取，与分文件数据配套。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 字段：`inputs__fixed_inputs_uint8`、`inputs__fixed_inputs_float32`、`inputs__labels_int64`、`inputs__source_indices_int64`、`inputs__validation_positions_int64`、`weights__conv_weight_float32`、`weights__conv_bias_float32`、`weights__fc_weight_float32`、`weights__fc_bias_float32`、`activations__conv_output_float32`、`activations__relu_output_float32`、`activations__pool_output_float32`、`activations__flatten_output_float32`、`activations__scores_float32`、`activations__predictions_int64`。 |
| [samples.csv](../reproduction/runs/baseline_v1/float_reference_v1/samples.csv) | 固定样本编号、标签与来源对应表。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [verification.json](../reproduction/runs/baseline_v1/float_reference_v1/verification.json) | 历史浮点参考一致性检查记录。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |

### reproduction/runs/baseline_v1/float_reference_v1/activations

| 文件 | 用途与使用者 |
|---|---|
| [conv_output_float32.npy](../reproduction/runs/baseline_v1/float_reference_v1/activations/conv_output_float32.npy) | 历史浮点卷积输出，B 分析量化前后差异。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 4, 26, 26]`，`float32`。 |
| [flatten_output_float32.npy](../reproduction/runs/baseline_v1/float_reference_v1/activations/flatten_output_float32.npy) | 历史浮点展平输出，B 复现使用。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 676]`，`float32`。 |
| [pool_output_float32.npy](../reproduction/runs/baseline_v1/float_reference_v1/activations/pool_output_float32.npy) | 历史浮点池化输出，B 复现使用。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 4, 13, 13]`，`float32`。 |
| [predictions_int64.npy](../reproduction/runs/baseline_v1/float_reference_v1/activations/predictions_int64.npy) | 分数 argmax 得到的预测标签，仅标签一致不能替代逐层一致检查。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10]`，`int64`。 |
| [relu_output_float32.npy](../reproduction/runs/baseline_v1/float_reference_v1/activations/relu_output_float32.npy) | 历史浮点 ReLU 输出，B 复现使用。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 4, 26, 26]`，`float32`。 |
| [scores_float32.npy](../reproduction/runs/baseline_v1/float_reference_v1/activations/scores_float32.npy) | 历史浮点分类分数，不能直接与整数分数要求逐值相等。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 10]`，`float32`。 |

### reproduction/runs/baseline_v1/float_reference_v1/inputs

| 文件 | 用途与使用者 |
|---|---|
| [fixed_inputs_float32.npy](../reproduction/runs/baseline_v1/float_reference_v1/inputs/fixed_inputs_float32.npy) | 浮点输入：uint8/255.0 并加通道维，不是当前整数模型的缩放规则。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 1, 28, 28]`，`float32`。 |
| [fixed_inputs_uint8.npy](../reproduction/runs/baseline_v1/float_reference_v1/inputs/fixed_inputs_uint8.npy) | 固定十张原始 uint8 输入，顺序对应数字 0～9。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 28, 28]`，`uint8`。 |
| [labels_int64.npy](../reproduction/runs/baseline_v1/float_reference_v1/inputs/labels_int64.npy) | 固定输入的真实数字标签。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10]`，`int64`。 |
| [source_indices_int64.npy](../reproduction/runs/baseline_v1/float_reference_v1/inputs/source_indices_int64.npy) | 样本在原 MNIST 训练数据池中的索引。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10]`，`int64`。 |
| [validation_positions_int64.npy](../reproduction/runs/baseline_v1/float_reference_v1/inputs/validation_positions_int64.npy) | 样本在固定验证子集中的位置。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10]`，`int64`。 |

### reproduction/runs/baseline_v1/float_reference_v1/manual_check

| 文件 | 用途与使用者 |
|---|---|
| [manual_conv.json](../reproduction/runs/baseline_v1/float_reference_v1/manual_check/manual_conv.json) | 指定卷积位置的手算示例和结果，帮助理解卷积。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |
| [manual_conv_terms.csv](../reproduction/runs/baseline_v1/float_reference_v1/manual_check/manual_conv_terms.csv) | 手算卷积的每项输入、权重及乘积明细。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 |

### reproduction/runs/baseline_v1/float_reference_v1/weights

| 文件 | 用途与使用者 |
|---|---|
| [conv_bias_float32.npy](../reproduction/runs/baseline_v1/float_reference_v1/weights/conv_bias_float32.npy) | 卷积层偏置，类型 float32。NumPy 单数组，供软件读取检查。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[4]`，`float32`。 |
| [conv_weight_float32.npy](../reproduction/runs/baseline_v1/float_reference_v1/weights/conv_weight_float32.npy) | 卷积层权重，类型 float32。NumPy 单数组，供软件读取检查。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[4, 1, 3, 3]`，`float32`。 |
| [fc_bias_float32.npy](../reproduction/runs/baseline_v1/float_reference_v1/weights/fc_bias_float32.npy) | 全连接层偏置，类型 float32。NumPy 单数组，供软件读取检查。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10]`，`float32`。 |
| [fc_weight_float32.npy](../reproduction/runs/baseline_v1/float_reference_v1/weights/fc_weight_float32.npy) | 全连接层权重，类型 float32。NumPy 单数组，供软件读取检查。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 676]`，`float32`。 |

## 仓库外围的配套文件

以下文件属于原有仓库布局，同样保持原位。

| 文件 | 作用 |
|---|---|
| [仓库 README](../../../README.md) | 全组导航。 |
| [.gitattributes](../../../.gitattributes) | 防止 Git 自动换行转换破坏交接文件哈希。 |
| [downloads/README.md](../../../downloads/README.md) | ZIP 下载与维护说明。 |
| [downloads/digitcnn_v1_int8_handoff_v1.zip](../../../downloads/digitcnn_v1_int8_handoff_v1.zip) | 完整模型文件夹快照，解压后结构与在线目录对应。 |
| [downloads/SHA256SUMS.txt](../../../downloads/SHA256SUMS.txt) | ZIP 校验值，检查下载完整性。 |
| [tools/package_digitcnn.py](../../../tools/package_digitcnn.py) | 同步整包清单、ZIP 和 ZIP 校验值的维护工具。 |

以上外围链接在仓库中有效；单独下载 ZIP 时外围文件不在解压目录内，包内链接仍可使用。

## 修改与验证约定

维护说明后，从仓库根目录执行 `python tools/package_digitcnn.py` 同步清单和 ZIP，再进入模型目录运行 `python verify_package.py`。模型参数或整数规则变更应另建版本，并重新验收精度与整数计算。

先做整包检查，再做数值验证。`verify_release.py` 和 `verify_packaging.py` 会写报告，环境变化可能使报告字节变化，不自动表示权重改变。B 重跑训练/实验时应使用独立副本或明确的新输出目录，保留交接版本供 A/C 对照。

本机可能出现 .ipynb_checkpoints、__pycache__ 等自动缓存，它们不属于 Git/ZIP 交付文件，也不是额外模型。
