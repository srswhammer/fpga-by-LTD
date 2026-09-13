"""维护交接包：生成中文逐文件索引、完整性清单和 ZIP。

从仓库根目录运行：python tools/package_digitcnn.py
新增文件时，在 DESCRIPTIONS 中补充中文用途；否则拒绝打包，避免出现无说明文件。
只更新交接说明和整包清单，不重新生成权重，也不替代数值验收。
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import zipfile

REPO = Path(__file__).resolve().parents[1]
SKIP = {"__pycache__", ".ipynb_checkpoints", ".venv", "work", "data"}
INDEX = "01_使用说明/逐文件索引.md"
MANIFEST = "04_验证与评估/整包校验清单.json"

# 每个交付文件的中文用途（与实际目录一一对应）。
DESCRIPTIONS = {
    "README.md": "唯一交接首页，只提供最常用入口。",
    "00_阅读地图.md": "中文总地图：角色路线、五层目录、数据关系和常见误解。",
    "01_使用说明/README.md": "文档目录的本地导航。",
    "01_使用说明/运行与环境.md": "从环境安装到运行、Notebook 和常见报错的操作指南。",
    "01_使用说明/A_硬件接手.md": "A 的交接清单、逐层验收流程与硬件常见错误。",
    "01_使用说明/C_软件联调.md": "C 的软件调用、摄像头预处理和待协商硬件接口。",
    "01_使用说明/数值与接口.md": "定点计算、参数布局和待确认通信协议的集中说明。",
    "01_使用说明/当前状态.md": "已完成、未完成和验收边界。",
    "02_运行代码/README.md": "运行代码阅读顺序及三个验证脚本的区别。",
    "02_运行代码/示例图片/README.md": "示例图片命名规则、来源和运行方法。",
    "03_模型数据/README.md": "当前模型数据的阅读顺序及固定版本约定。",
    "03_模型数据/参数/README.md": "权重命名、不同存储格式与对应使用者。",
    "03_模型数据/逐层标准答案/README.md": "逐层数组的计算顺序、含义及形状。",
    "04_验证与评估/README.md": "评估结果、历史记录、本次验收和整包校验的区分。",
    "04_验证与评估/精度结果/README.md": "评估文件前缀/后缀的含义和快速查看入口。",
    "04_验证与评估/原验收记录/README.md": "三个原验收 JSON 的时间范围与含义。",
    "05_训练复现/README.md": "B 的实验入口、来源与源码保留规则。",
    "05_训练复现/实验工程/README.md": "历史实验运行顺序、写入位置和避免覆盖的方法。",
    "05_训练复现/代码导读.md": "训练各脚本关系、原 model.py 中文解释和调参关联。",
    "05_训练复现/历史设计说明/README.md": "历史文档阅读边界，避免把旧路径当当前入口。",
    "02_运行代码/演示.ipynb": "四步中文 Notebook：环境、固定输入、单图预处理、评估图表。",
    "启动.py": "唯一命令行入口：演示、校验、图片；出错即停止。",
    ".gitignore": "排除缓存、虚拟环境、数据下载和 work 新报告。",
    "02_运行代码/paths.py": "统一路径表，定义中文目录及 work 报告位置。",
    "04_验证与评估/整包校验清单.json": "打包生成的全文件 SHA-256，不包含清单自身。",
    "04_验证与评估/目录整理验收.json": "路径迁移、核心结果不变、独立检查及 Notebook 执行的复核报告。",
    "01_使用说明/逐文件索引.md": "逐文件中文解释、可点击路径及全部旧文件去向。",
    "02_运行代码/demo.py": "命令行预测演示，默认跑固定十张，--image 可指定图片，C 首次运行。",
    "05_训练复现/历史设计说明/定点候选推导.md": "定点候选阶段设计推导历史，B 解释数值来源时查阅。",
    "05_训练复现/历史设计说明/浮点参考格式.md": "历史浮点参考格式，与 reproduction 中浮点数据配套阅读。",
    "05_训练复现/历史设计说明/原整数验收说明.md": "整数第一版的软件验收说明；当前数值以 03_模型数据/config.json 为准。",
    "05_训练复现/历史设计说明/基线模型规格.md": "模型结构和基线设计规格，A/B 理解各层及参数时参考。",
    "02_运行代码/示例图片/camera_like_00_label_1.png": "合成拍照风格示例，真实标签为 1；C 测预处理/预测，不是 OV5640 实拍数据。",
    "02_运行代码/示例图片/camera_like_01_label_4.png": "合成拍照风格示例，真实标签为 4；C 测预处理/预测，不是 OV5640 实拍数据。",
    "02_运行代码/示例图片/camera_like_02_label_7.png": "合成拍照风格示例，真实标签为 7；C 测预处理/预测，不是 OV5640 实拍数据。",
    "02_运行代码/示例图片/camera_like_03_label_4.png": "合成拍照风格示例，真实标签为 4；C 测预处理/预测，不是 OV5640 实拍数据。",
    "02_运行代码/示例图片/camera_like_04_label_7.png": "合成拍照风格示例，真实标签为 7；C 测预处理/预测，不是 OV5640 实拍数据。",
    "02_运行代码/示例图片/camera_like_05_label_7.png": "合成拍照风格示例，真实标签为 7；C 测预处理/预测，不是 OV5640 实拍数据。",
    "02_运行代码/示例图片/camera_like_06_label_7.png": "合成拍照风格示例，真实标签为 7；C 测预处理/预测，不是 OV5640 实拍数据。",
    "02_运行代码/示例图片/camera_like_07_label_2.png": "合成拍照风格示例，真实标签为 2；C 测预处理/预测，不是 OV5640 实拍数据。",
    "02_运行代码/示例图片/camera_like_08_label_5.png": "合成拍照风格示例，真实标签为 5；C 测预处理/预测，不是 OV5640 实拍数据。",
    "02_运行代码/示例图片/camera_like_09_label_2.png": "合成拍照风格示例，真实标签为 2；C 测预处理/预测，不是 OV5640 实拍数据。",
    "02_运行代码/示例图片/camera_like_10_label_6.png": "合成拍照风格示例，真实标签为 6；C 测预处理/预测，不是 OV5640 实拍数据。",
    "02_运行代码/示例图片/camera_like_11_label_7.png": "合成拍照风格示例，真实标签为 7；C 测预处理/预测，不是 OV5640 实拍数据。",
    "02_运行代码/integer_inference.py": "带六阶段中文注释的 CPU 整数模型；默认读取 03_模型数据。",
    "04_验证与评估/原验收记录/packaging_checks.json": "上次独立整数验证结果与环境记录，不是 FPGA 验证报告。",
    "02_运行代码/preprocess.py": "单数字图片转 uint8[28,28]，处理灰度、背景、裁剪、缩放、居中；C 接摄像头 ROI 后调用。",
    "04_验证与评估/原验收记录/preprocessing_parity.json": "打包时原版与便携预处理的 22 张图片像素一致性记录。",
    "03_模型数据/config.json": "当前整数模型结构、位宽、缩放和数值规则，A 实现硬件的首要依据。",
    "04_验证与评估/精度结果/accuracy_comparison.png": "浮点/整数在三种数据上的精度对比图，组长/B 汇报使用。",
    "04_验证与评估/精度结果/official_test_confusion_int8.csv": "官方测试集 10000 张的整数混淆矩阵，B 分析错误分布。",
    "04_验证与评估/精度结果/official_test_confusion_int8.png": "官方测试集的整数混淆矩阵图，B 查看易混数字。",
    "04_验证与评估/精度结果/official_test_per_class_accuracy.csv": "官方测试集 10000 张各数字分类准确率，B 查薄弱类别。",
    "04_验证与评估/精度结果/official_test_predictions.npz": "官方测试集 10000 张逐样本预测相关数组，B 重算统计及定位分歧。 字段：`labels`、`float_predictions`、`float_scores`、`int8_predictions`、`int8_scores`。",
    "04_验证与评估/精度结果/official_test_quantization_disagreements.png": "量化前后预测有分歧的图片示例，B 分析误差。",
    "04_验证与评估/精度结果/summary.json": "验证集、官方测试集、合成拍照图的 float/int8 精度、饱和情况和验收汇总。",
    "04_验证与评估/精度结果/synthetic_camera_like_confusion_int8.csv": "合成拍照图 1000 张（不是实拍）的整数混淆矩阵，B 分析错误分布。",
    "04_验证与评估/精度结果/synthetic_camera_like_per_class_accuracy.csv": "合成拍照图 1000 张（不是实拍）各数字分类准确率，B 查薄弱类别。",
    "04_验证与评估/精度结果/synthetic_camera_like_predictions.npz": "合成拍照图 1000 张（不是实拍）逐样本预测相关数组，B 重算统计及定位分歧。 字段：`labels`、`float_predictions`、`float_scores`、`int8_predictions`、`int8_scores`。",
    "04_验证与评估/精度结果/validation_confusion_int8.csv": "验证集 5000 张的整数混淆矩阵，B 分析错误分布。",
    "04_验证与评估/精度结果/validation_per_class_accuracy.csv": "验证集 5000 张各数字分类准确率，B 查薄弱类别。",
    "04_验证与评估/精度结果/validation_predictions.npz": "验证集 5000 张逐样本预测相关数组，B 重算统计及定位分歧。 字段：`labels`、`float_predictions`、`float_scores`、`int8_predictions`、`int8_scores`。",
    "03_模型数据/manifest.json": "发布数据的元数据与哈希清单，供 02_运行代码/verify_release.py 核验，范围不同于整包清单。",
    "03_模型数据/参数/binary_layout.json": "BIN 各参数段的名称、偏移、长度和形状，解析 BIN 必须参考。",
    "03_模型数据/参数/conv_bias_int32.mem": "卷积层偏置，类型 int32。十六进制补码文本，每行一个展平数据，供 A 仿真初始化。",
    "03_模型数据/参数/conv_bias_int32.npy": "卷积层偏置，类型 int32。NumPy 单数组，供软件读取检查。 数组：`[4]`，`int32`。",
    "03_模型数据/参数/conv_weight_int8.mem": "卷积层权重，类型 int8。十六进制补码文本，每行一个展平数据，供 A 仿真初始化。",
    "03_模型数据/参数/conv_weight_int8.npy": "卷积层权重，类型 int8。NumPy 单数组，供软件读取检查。 数组：`[4, 1, 3, 3]`，`int8`。",
    "03_模型数据/参数/fc_bias_int32.mem": "全连接层偏置，类型 int32。十六进制补码文本，每行一个展平数据，供 A 仿真初始化。",
    "03_模型数据/参数/fc_bias_int32.npy": "全连接层偏置，类型 int32。NumPy 单数组，供软件读取检查。 数组：`[10]`，`int32`。",
    "03_模型数据/参数/fc_weight_int8.mem": "全连接层权重，类型 int8。十六进制补码文本，每行一个展平数据，供 A 仿真初始化。",
    "03_模型数据/参数/fc_weight_int8.npy": "全连接层权重，类型 int8。NumPy 单数组，供软件读取检查。 数组：`[10, 676]`，`int8`。",
    "03_模型数据/参数/model_parameters_int.npz": "四组整数参数的合并存储，当前 IntegerDigitCNN 默认加载它。 字段：`conv_weight_int8`、`conv_bias_int32`、`fc_weight_int8`、`fc_bias_int32`。",
    "03_模型数据/参数/model_parameters_v1.bin": "同一套参数的无头字节，共 6852 字节，int32 为小端；供硬件加载方案使用。",
    "03_模型数据/逐层标准答案/conv_acc_int32.npy": "卷积乘加并加偏置的 int32 输出，未 ReLU/缩放，A 第一阶段逐值对比。 数组：`[10, 4, 26, 26]`，`int32`。",
    "03_模型数据/逐层标准答案/fixed_inputs_uint8.npy": "固定十张原始 uint8 输入，顺序对应数字 0～9。 数组：`[10, 28, 28]`，`uint8`。",
    "03_模型数据/逐层标准答案/flatten_uint8.npy": "按通道、行、列展平的 676 维特征，A 检查 FC 输入顺序。 数组：`[10, 676]`，`uint8`。",
    "03_模型数据/逐层标准答案/labels_int64.npy": "固定输入的真实数字标签。 数组：`[10]`，`int64`。",
    "03_模型数据/逐层标准答案/pool_uint8.npy": "2×2 最大池化特征，A 检查窗口和通道顺序。 数组：`[10, 4, 13, 13]`，`uint8`。",
    "03_模型数据/逐层标准答案/predictions_int64.npy": "分数 argmax 得到的预测标签，仅标签一致不能替代逐层一致检查。 数组：`[10]`，`int64`。",
    "03_模型数据/逐层标准答案/relu_uint8.npy": "ReLU、舍入右移和饱和后的 uint8 特征，A 检查数值转换。 数组：`[10, 4, 26, 26]`，`uint8`。",
    "03_模型数据/逐层标准答案/scores_int32.npy": "每图十个有符号 int32 分数，A/C 对比最终数值，不是概率。 数组：`[10, 10]`，`int32`。",
    "04_验证与评估/原验收记录/verification.json": "整理前发布验证结果，保留原字节作为历史证据。",
    "05_训练复现/实验工程/design_fixed_point_candidate.py": "分析浮点范围、设计定点缩放并生成候选整数参数。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/evaluate_baseline.py": "加载浮点 checkpoint，在 MNIST 测试集评估并输出统计/图表。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/evaluate_integer_v1.py": "在实验工程内生成 runs/digitcnn_v1_int8，包含精度评估和参数导出；不会自动更新交接包 03_模型数据。",
    "05_训练复现/实验工程/export_float_reference.py": "导出浮点权重、固定输入、逐层结果和手算卷积示例。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/integer_inference.py": "实验阶段的整数推理实现，默认读取本实验工程的 runs/digitcnn_v1_int8；A/C 使用 02_运行代码中的同名文件。",
    "05_训练复现/实验工程/model.py": "原始 PyTorch CNN 结构，checkpoint 记录其哈希，复现时保持原字节。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/predict.py": "原含 PyTorch 的图片预处理和浮点预测，用于 B 复现；C 使用 02_运行代码中的便携预处理。",
    "05_训练复现/实验工程/requirements.lock.txt": "原训练电脑完整依赖版本记录，供 B 追溯，不要直接用来升级 PYNQ。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/best_model.pt": "最佳浮点 PyTorch checkpoint，B 复现使用，不能直接加载到 FPGA。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/config.json": "本次浮点基线训练配置与超参数。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/fixed_point_candidate_v1/accuracy_evaluation.json": "候选精度评估记录，当前验收看 04_验证与评估/精度结果/summary.json。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/fixed_point_candidate_v1/candidate_config.json": "历史候选定点位宽和缩放配置。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/fixed_point_candidate_v1/candidate_parameters_unvalidated.npz": "候选参数快照，名字保留当时状态；硬件接手采用 03_模型数据/参数。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 字段：`conv_weight_int8`、`conv_bias_int32`、`fc_weight_int8`、`fc_bias_int32`。",
    "05_训练复现/实验工程/runs/baseline_v1/fixed_point_candidate_v1/integer_rule_example.json": "候选整数运算规则的具体示例。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/fixed_point_candidate_v1/range_analysis.json": "范围分析结果，B 理解缩放与溢出风险。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/fixed_point_candidate_v1/verification.json": "候选阶段数值一致性检查记录。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/activations/conv_output_float32.npy": "历史浮点卷积输出，B 分析量化前后差异。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 4, 26, 26]`，`float32`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/activations/flatten_output_float32.npy": "历史浮点展平输出，B 复现使用。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 676]`，`float32`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/activations/pool_output_float32.npy": "历史浮点池化输出，B 复现使用。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 4, 13, 13]`，`float32`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/activations/predictions_int64.npy": "分数 argmax 得到的预测标签，仅标签一致不能替代逐层一致检查。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10]`，`int64`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/activations/relu_output_float32.npy": "历史浮点 ReLU 输出，B 复现使用。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 4, 26, 26]`，`float32`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/activations/scores_float32.npy": "历史浮点分类分数，不能直接与整数分数要求逐值相等。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 10]`，`float32`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/fixed_inputs_overview.png": "历史浮点参考固定十张输入总览。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/inputs/fixed_inputs_float32.npy": "浮点输入：uint8/255.0 并加通道维，不是当前整数模型的缩放规则。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 1, 28, 28]`，`float32`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/inputs/fixed_inputs_uint8.npy": "固定十张原始 uint8 输入，顺序对应数字 0～9。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 28, 28]`，`uint8`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/inputs/labels_int64.npy": "固定输入的真实数字标签。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10]`，`int64`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/inputs/source_indices_int64.npy": "样本在原 MNIST 训练数据池中的索引。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10]`，`int64`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/inputs/validation_positions_int64.npy": "样本在固定验证子集中的位置。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10]`，`int64`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/manifest.json": "浮点参考来源、形状、布局、哈希与选样规则。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/manual_check/manual_conv.json": "指定卷积位置的手算示例和结果，帮助理解卷积。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/manual_check/manual_conv_terms.csv": "手算卷积的每项输入、权重及乘积明细。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/reference_bundle_v1.npz": "历史浮点参考数组合并存储，便于一次读取，与分文件数据配套。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 字段：`inputs__fixed_inputs_uint8`、`inputs__fixed_inputs_float32`、`inputs__labels_int64`、`inputs__source_indices_int64`、`inputs__validation_positions_int64`、`weights__conv_weight_float32`、`weights__conv_bias_float32`、`weights__fc_weight_float32`、`weights__fc_bias_float32`、`activations__conv_output_float32`、`activations__relu_output_float32`、`activations__pool_output_float32`、`activations__flatten_output_float32`、`activations__scores_float32`、`activations__predictions_int64`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/samples.csv": "固定样本编号、标签与来源对应表。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/verification.json": "历史浮点参考一致性检查记录。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/weights/conv_bias_float32.npy": "卷积层偏置，类型 float32。NumPy 单数组，供软件读取检查。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[4]`，`float32`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/weights/conv_weight_float32.npy": "卷积层权重，类型 float32。NumPy 单数组，供软件读取检查。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[4, 1, 3, 3]`，`float32`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/weights/fc_bias_float32.npy": "全连接层偏置，类型 float32。NumPy 单数组，供软件读取检查。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10]`，`float32`。",
    "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/weights/fc_weight_float32.npy": "全连接层权重，类型 float32。NumPy 单数组，供软件读取检查。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 数组：`[10, 676]`，`float32`。",
    "05_训练复现/实验工程/runs/baseline_v1/history.csv": "逐轮训练/验证指标，B 查看学习过程。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/reload_check.npz": "checkpoint 重新加载对照数组，检查保存前后行为。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 字段：`inputs`、`labels`、`expected_scores`。",
    "05_训练复现/实验工程/runs/baseline_v1/smoke_check.json": "基线训练的小规模运行检查记录。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/runs/baseline_v1/split_indices.npz": "固定训练/验证划分索引，保证复现使用同一划分。 【B 的历史/复现材料；A/C 初期可暂不阅读。】 字段：`train`、`validation`。",
    "05_训练复现/实验工程/runs/baseline_v1/summary.json": "浮点基线训练结果摘要。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/train_baseline.py": "训练基线并保存最佳 checkpoint，B 重训时另设 run-dir。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/validate_software_prediction.py": "验证原始图片预测流程，生成合成示例与检查记录，需要 MNIST 数据。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/verify_fixed_point_candidate.py": "检查候选整数运算和参考结果一致性。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/verify_float_reference.py": "核验历史浮点导出与 checkpoint/逐层结果一致性。 【B 的历史/复现材料；A/C 初期可暂不阅读。】",
    "05_训练复现/实验工程/verify_integer_v1.py": "检查实验工程内生成的 runs/digitcnn_v1_int8；当前交接包校验从根目录的启动.py运行。",
    "02_运行代码/requirements-notebook.txt": "增加 JupyterLab/ipykernel 的依赖清单，C 打开 Notebook 时使用。",
    "02_运行代码/requirements.txt": "运行整数模型及预处理的 NumPy/Pillow 依赖范围，A/C 安装使用。",
    "02_运行代码/verify_package.py": "核验整包清单，纯标准库，不改发布文件。",
    "02_运行代码/verify_packaging.py": "14 张独立标量检查及 BIN/MEM 解码对比，新报告写 work/独立整数验证.json。",
    "02_运行代码/verify_release.py": "发布参数、逐层答案和精度统计复核，新报告写 work/发布验证.json。"
}

# 覆盖旧交接目录后，读者可查到每个旧文件的新位置。
OLD_TO_NEW = {
    "demo.py": "02_运行代码/demo.py",
    "docs/fixed_point_candidate_v1.md": "05_训练复现/历史设计说明/定点候选推导.md",
    "docs/float_reference_format_v1.md": "05_训练复现/历史设计说明/浮点参考格式.md",
    "docs/integer_release_v1.md": "05_训练复现/历史设计说明/原整数验收说明.md",
    "docs/model_spec_v1.md": "05_训练复现/历史设计说明/基线模型规格.md",
    "examples/camera_like_00_label_1.png": "02_运行代码/示例图片/camera_like_00_label_1.png",
    "examples/camera_like_01_label_4.png": "02_运行代码/示例图片/camera_like_01_label_4.png",
    "examples/camera_like_02_label_7.png": "02_运行代码/示例图片/camera_like_02_label_7.png",
    "examples/camera_like_03_label_4.png": "02_运行代码/示例图片/camera_like_03_label_4.png",
    "examples/camera_like_04_label_7.png": "02_运行代码/示例图片/camera_like_04_label_7.png",
    "examples/camera_like_05_label_7.png": "02_运行代码/示例图片/camera_like_05_label_7.png",
    "examples/camera_like_06_label_7.png": "02_运行代码/示例图片/camera_like_06_label_7.png",
    "examples/camera_like_07_label_2.png": "02_运行代码/示例图片/camera_like_07_label_2.png",
    "examples/camera_like_08_label_5.png": "02_运行代码/示例图片/camera_like_08_label_5.png",
    "examples/camera_like_09_label_2.png": "02_运行代码/示例图片/camera_like_09_label_2.png",
    "examples/camera_like_10_label_6.png": "02_运行代码/示例图片/camera_like_10_label_6.png",
    "examples/camera_like_11_label_7.png": "02_运行代码/示例图片/camera_like_11_label_7.png",
    "integer_inference.py": "02_运行代码/integer_inference.py",
    "packaging_checks.json": "04_验证与评估/原验收记录/packaging_checks.json",
    "preprocess.py": "02_运行代码/preprocess.py",
    "preprocessing_parity.json": "04_验证与评估/原验收记录/preprocessing_parity.json",
    "release/config.json": "03_模型数据/config.json",
    "release/evaluation/accuracy_comparison.png": "04_验证与评估/精度结果/accuracy_comparison.png",
    "release/evaluation/official_test_confusion_int8.csv": "04_验证与评估/精度结果/official_test_confusion_int8.csv",
    "release/evaluation/official_test_confusion_int8.png": "04_验证与评估/精度结果/official_test_confusion_int8.png",
    "release/evaluation/official_test_per_class_accuracy.csv": "04_验证与评估/精度结果/official_test_per_class_accuracy.csv",
    "release/evaluation/official_test_predictions.npz": "04_验证与评估/精度结果/official_test_predictions.npz",
    "release/evaluation/official_test_quantization_disagreements.png": "04_验证与评估/精度结果/official_test_quantization_disagreements.png",
    "release/evaluation/summary.json": "04_验证与评估/精度结果/summary.json",
    "release/evaluation/synthetic_camera_like_confusion_int8.csv": "04_验证与评估/精度结果/synthetic_camera_like_confusion_int8.csv",
    "release/evaluation/synthetic_camera_like_per_class_accuracy.csv": "04_验证与评估/精度结果/synthetic_camera_like_per_class_accuracy.csv",
    "release/evaluation/synthetic_camera_like_predictions.npz": "04_验证与评估/精度结果/synthetic_camera_like_predictions.npz",
    "release/evaluation/validation_confusion_int8.csv": "04_验证与评估/精度结果/validation_confusion_int8.csv",
    "release/evaluation/validation_per_class_accuracy.csv": "04_验证与评估/精度结果/validation_per_class_accuracy.csv",
    "release/evaluation/validation_predictions.npz": "04_验证与评估/精度结果/validation_predictions.npz",
    "release/manifest.json": "03_模型数据/manifest.json",
    "release/parameters/binary_layout.json": "03_模型数据/参数/binary_layout.json",
    "release/parameters/conv_bias_int32.mem": "03_模型数据/参数/conv_bias_int32.mem",
    "release/parameters/conv_bias_int32.npy": "03_模型数据/参数/conv_bias_int32.npy",
    "release/parameters/conv_weight_int8.mem": "03_模型数据/参数/conv_weight_int8.mem",
    "release/parameters/conv_weight_int8.npy": "03_模型数据/参数/conv_weight_int8.npy",
    "release/parameters/fc_bias_int32.mem": "03_模型数据/参数/fc_bias_int32.mem",
    "release/parameters/fc_bias_int32.npy": "03_模型数据/参数/fc_bias_int32.npy",
    "release/parameters/fc_weight_int8.mem": "03_模型数据/参数/fc_weight_int8.mem",
    "release/parameters/fc_weight_int8.npy": "03_模型数据/参数/fc_weight_int8.npy",
    "release/parameters/model_parameters_int.npz": "03_模型数据/参数/model_parameters_int.npz",
    "release/parameters/model_parameters_v1.bin": "03_模型数据/参数/model_parameters_v1.bin",
    "release/reference/conv_acc_int32.npy": "03_模型数据/逐层标准答案/conv_acc_int32.npy",
    "release/reference/fixed_inputs_uint8.npy": "03_模型数据/逐层标准答案/fixed_inputs_uint8.npy",
    "release/reference/flatten_uint8.npy": "03_模型数据/逐层标准答案/flatten_uint8.npy",
    "release/reference/labels_int64.npy": "03_模型数据/逐层标准答案/labels_int64.npy",
    "release/reference/pool_uint8.npy": "03_模型数据/逐层标准答案/pool_uint8.npy",
    "release/reference/predictions_int64.npy": "03_模型数据/逐层标准答案/predictions_int64.npy",
    "release/reference/relu_uint8.npy": "03_模型数据/逐层标准答案/relu_uint8.npy",
    "release/reference/scores_int32.npy": "03_模型数据/逐层标准答案/scores_int32.npy",
    "release/verification.json": "04_验证与评估/原验收记录/verification.json",
    "reproduction/design_fixed_point_candidate.py": "05_训练复现/实验工程/design_fixed_point_candidate.py",
    "reproduction/evaluate_baseline.py": "05_训练复现/实验工程/evaluate_baseline.py",
    "reproduction/evaluate_integer_v1.py": "05_训练复现/实验工程/evaluate_integer_v1.py",
    "reproduction/export_float_reference.py": "05_训练复现/实验工程/export_float_reference.py",
    "reproduction/integer_inference.py": "05_训练复现/实验工程/integer_inference.py",
    "reproduction/model.py": "05_训练复现/实验工程/model.py",
    "reproduction/predict.py": "05_训练复现/实验工程/predict.py",
    "reproduction/requirements.lock.txt": "05_训练复现/实验工程/requirements.lock.txt",
    "reproduction/runs/baseline_v1/best_model.pt": "05_训练复现/实验工程/runs/baseline_v1/best_model.pt",
    "reproduction/runs/baseline_v1/config.json": "05_训练复现/实验工程/runs/baseline_v1/config.json",
    "reproduction/runs/baseline_v1/fixed_point_candidate_v1/accuracy_evaluation.json": "05_训练复现/实验工程/runs/baseline_v1/fixed_point_candidate_v1/accuracy_evaluation.json",
    "reproduction/runs/baseline_v1/fixed_point_candidate_v1/candidate_config.json": "05_训练复现/实验工程/runs/baseline_v1/fixed_point_candidate_v1/candidate_config.json",
    "reproduction/runs/baseline_v1/fixed_point_candidate_v1/candidate_parameters_unvalidated.npz": "05_训练复现/实验工程/runs/baseline_v1/fixed_point_candidate_v1/candidate_parameters_unvalidated.npz",
    "reproduction/runs/baseline_v1/fixed_point_candidate_v1/integer_rule_example.json": "05_训练复现/实验工程/runs/baseline_v1/fixed_point_candidate_v1/integer_rule_example.json",
    "reproduction/runs/baseline_v1/fixed_point_candidate_v1/range_analysis.json": "05_训练复现/实验工程/runs/baseline_v1/fixed_point_candidate_v1/range_analysis.json",
    "reproduction/runs/baseline_v1/fixed_point_candidate_v1/verification.json": "05_训练复现/实验工程/runs/baseline_v1/fixed_point_candidate_v1/verification.json",
    "reproduction/runs/baseline_v1/float_reference_v1/activations/conv_output_float32.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/activations/conv_output_float32.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/activations/flatten_output_float32.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/activations/flatten_output_float32.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/activations/pool_output_float32.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/activations/pool_output_float32.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/activations/predictions_int64.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/activations/predictions_int64.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/activations/relu_output_float32.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/activations/relu_output_float32.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/activations/scores_float32.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/activations/scores_float32.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/fixed_inputs_overview.png": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/fixed_inputs_overview.png",
    "reproduction/runs/baseline_v1/float_reference_v1/inputs/fixed_inputs_float32.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/inputs/fixed_inputs_float32.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/inputs/fixed_inputs_uint8.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/inputs/fixed_inputs_uint8.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/inputs/labels_int64.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/inputs/labels_int64.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/inputs/source_indices_int64.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/inputs/source_indices_int64.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/inputs/validation_positions_int64.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/inputs/validation_positions_int64.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/manifest.json": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/manifest.json",
    "reproduction/runs/baseline_v1/float_reference_v1/manual_check/manual_conv.json": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/manual_check/manual_conv.json",
    "reproduction/runs/baseline_v1/float_reference_v1/manual_check/manual_conv_terms.csv": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/manual_check/manual_conv_terms.csv",
    "reproduction/runs/baseline_v1/float_reference_v1/reference_bundle_v1.npz": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/reference_bundle_v1.npz",
    "reproduction/runs/baseline_v1/float_reference_v1/samples.csv": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/samples.csv",
    "reproduction/runs/baseline_v1/float_reference_v1/verification.json": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/verification.json",
    "reproduction/runs/baseline_v1/float_reference_v1/weights/conv_bias_float32.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/weights/conv_bias_float32.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/weights/conv_weight_float32.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/weights/conv_weight_float32.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/weights/fc_bias_float32.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/weights/fc_bias_float32.npy",
    "reproduction/runs/baseline_v1/float_reference_v1/weights/fc_weight_float32.npy": "05_训练复现/实验工程/runs/baseline_v1/float_reference_v1/weights/fc_weight_float32.npy",
    "reproduction/runs/baseline_v1/history.csv": "05_训练复现/实验工程/runs/baseline_v1/history.csv",
    "reproduction/runs/baseline_v1/reload_check.npz": "05_训练复现/实验工程/runs/baseline_v1/reload_check.npz",
    "reproduction/runs/baseline_v1/smoke_check.json": "05_训练复现/实验工程/runs/baseline_v1/smoke_check.json",
    "reproduction/runs/baseline_v1/split_indices.npz": "05_训练复现/实验工程/runs/baseline_v1/split_indices.npz",
    "reproduction/runs/baseline_v1/summary.json": "05_训练复现/实验工程/runs/baseline_v1/summary.json",
    "reproduction/train_baseline.py": "05_训练复现/实验工程/train_baseline.py",
    "reproduction/validate_software_prediction.py": "05_训练复现/实验工程/validate_software_prediction.py",
    "reproduction/verify_fixed_point_candidate.py": "05_训练复现/实验工程/verify_fixed_point_candidate.py",
    "reproduction/verify_float_reference.py": "05_训练复现/实验工程/verify_float_reference.py",
    "reproduction/verify_integer_v1.py": "05_训练复现/实验工程/verify_integer_v1.py",
    "requirements-notebook.txt": "02_运行代码/requirements-notebook.txt",
    "requirements.txt": "02_运行代码/requirements.txt",
    "verify_package.py": "02_运行代码/verify_package.py",
    "verify_packaging.py": "02_运行代码/verify_packaging.py",
    "verify_release.py": "02_运行代码/verify_release.py",
    ".gitignore": ".gitignore",
    "README.md": "README.md",
    "PACKAGE_MANIFEST.json": "04_验证与评估/整包校验清单.json",
    "demo.ipynb": "02_运行代码/演示.ipynb",
    "docs/FILE_GUIDE.md": "01_使用说明/逐文件索引.md",
    "docs/FOR_A.md": "01_使用说明/A_硬件接手.md",
    "docs/FOR_C.md": "01_使用说明/C_软件联调.md",
    "docs/STATUS.md": "01_使用说明/当前状态.md",
    "docs/REPRODUCIBILITY.md": "05_训练复现/README.md",
    "docs/hardware_interface_draft_v1.md": "01_使用说明/数值与接口.md"
}

def list_files(source):
    """只选实际交付文件，不把运行缓存、下载数据或新实验发布产物放进 ZIP。"""
    return sorted(p for p in source.rglob("*") if p.is_file()
                  and not SKIP.intersection(p.relative_to(source).parts)
                  and "05_训练复现/实验工程/runs/digitcnn_v1_int8/" not in p.relative_to(source).as_posix()
                  and p.suffix != ".pyc")


def build(source, downloads):
    source = source.resolve()
    downloads.mkdir(parents=True, exist_ok=True)
    names = {p.relative_to(source).as_posix() for p in list_files(source)} | {INDEX, MANIFEST}
    missing = names - DESCRIPTIONS.keys()
    if missing:
        raise ValueError("以下文件缺少中文用途，请先维护 DESCRIPTIONS：" + repr(sorted(missing)))
    stale = DESCRIPTIONS.keys() - names
    if stale:
        raise ValueError("用途表引用了不存在的文件：" + repr(sorted(stale)))

    # 文件地图由实际目录生成，每个交付文件只有一条用途记录。
    lines = ["# 逐文件索引与旧路径去向\n\n",
             "先看 [总地图](../00_阅读地图.md) 选择角色；本页用于按具体文件名查用途。\n\n",
             f"当前共 {len(names)} 个交付文件（包含说明和清单），按所在目录分组。点击文件名可查看。\n\n"]
    directories = sorted({str(Path(n).parent).replace('\\', '/') for n in names}, key=lambda n: (n != '.', n))
    for folder in directories:
        label = "包根目录" if folder == "." else folder
        lines += [f"## {label}\n\n", "| 文件 | 中文用途 |\n|---|---|\n"]
        for name in sorted(names):
            if Path(name).parent.as_posix() != folder:
                continue
            link = Path(os.path.relpath(source/name, (source/INDEX).parent)).as_posix()
            desc = DESCRIPTIONS[name].replace("|", "\\|")
            lines.append(f"| [{Path(name).name}]({link}) | {desc} |\n")
        lines.append("\n")

    lines += ["## 旧交接包中的文件现在在哪里\n\n",
              "旧路径已被本版替换。参数和固定数组迁移时保留原值；角色指南重写，旧文件总说明由本页替代。\n\n",
              "| 旧相对路径 | 新相对路径 |\n|---|---|\n"]
    for old, new in sorted(OLD_TO_NEW.items()):
        if new not in names:
            raise ValueError("迁移目标不存在：" + new)
        link = Path(os.path.relpath(source/new, (source/INDEX).parent)).as_posix()
        lines.append(f"| `{old}` | [{new}]({link}) |\n")
    (source/INDEX).write_text(''.join(lines), encoding="utf-8", newline="\n")

    # 检查所有 Markdown 本地链接，目录重构后不能留下失效入口。
    for path in list_files(source):
        if path.suffix != '.md':
            continue
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):
                continue
            target = (path.parent/link.split('#')[0]).resolve()
            if target == (source/MANIFEST).resolve():
                continue  # 下一步生成清单。
            if not target.exists():
                raise ValueError(f"文档链接失效：{path.relative_to(source)} -> {link}")

    files = [p for p in list_files(source) if p.relative_to(source).as_posix() != MANIFEST]
    manifest = {"package": "digitcnn_v1_int8", "layout_version": "handoff_layout_v2", "files": {
        p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    manifest_path = source/MANIFEST
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')

    # 使用固定时间戳和排序。同样文件字节产生同样 ZIP，便于比较交付内容。
    archive = downloads/'digitcnn_v1_int8_handoff_v1.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(files+[manifest_path]):
            info = zipfile.ZipInfo('digitcnn_v1_int8/'+path.relative_to(source).as_posix(), (2026,9,13,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (downloads/'SHA256SUMS.txt').write_text(f'{sha}  {archive.name}\n', encoding='ascii', newline='\n')
    print(f"交接包已更新：{len(files)+1} 个文件，{archive.stat().st_size} 字节")
    print("ZIP SHA-256：", sha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='更新中文交接索引和 ZIP')
    parser.add_argument('--source', type=Path, default=REPO/'models/digitcnn_v1_int8')
    parser.add_argument('--output', type=Path, default=REPO/'downloads')
    args = parser.parse_args()
    build(args.source, args.output)
