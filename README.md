# FPGA 三人协作项目

## CNN 模型第一版

进入 **[digitcnn_v1_int8 交接首页](models/digitcnn_v1_int8/README.md)** 查看运行命令、测试结果和 A/C 接手材料。

| 入口 | 内容 |
|---|---|
| [逐文件用途说明](models/digitcnn_v1_int8/docs/FILE_GUIDE.md) | 按原目录解释全部文件，标明 A/C 阅读顺序 |
| [A：硬件接手](models/digitcnn_v1_int8/docs/FOR_A.md) | 定点规则、参数、逐层答案和接口协商项 |
| [C：Notebook 与联调](models/digitcnn_v1_int8/docs/FOR_C.md) | 图片预处理、纯整数调用、摄像头 ROI 要求 |
| [在线查看 Notebook](models/digitcnn_v1_int8/demo.ipynb) | 可复现的软件演示 |
| [ZIP 下载](downloads/README.md) | 整包及 SHA-256 |
| [状态与下一步](models/digitcnn_v1_int8/docs/STATUS.md) | 软件已验证；FPGA 与实拍待验证 |

第一版 MNIST 测试准确率 **95.21%**，参数原始大小 **6852 字节**。尚无板上性能或资源结论。

## 团队资料

- [开发进度](进度记录/开发进度记录.md)
- [问题记录](question.md)
- [创新想法](idea/innovations_collection.md)
- [板卡相关资料](资料/)
