# 模型交接下载

- **[下载 digitcnn_v1_int8 完整交接 ZIP](digitcnn_v1_int8_handoff_v1.zip)**（GitHub 文件页点击 Download raw file）。
- [SHA-256 校验值](SHA256SUMS.txt)。
- [先在线阅读交接首页](../models/digitcnn_v1_int8/README.md)。

解压后进入 `digitcnn_v1_int8`，先看 README。A 看 `docs/FOR_A.md`，C 看 `docs/FOR_C.md`，B 的训练复现材料在 `reproduction/`。一个完整包包含各角色入口，避免多人使用不同版本的权重。

ZIP 不包含 .venv、完整 MNIST 数据集、Jupyter 配置或日志。整包完整性检查无需额外安装库：`python verify_package.py`；安装 requirements 后可运行模型验证和演示。

维护者从仓库根目录执行 `python tools/package_digitcnn.py` 重新打包。对参数/数值规则的实质更改应建立新版本，并重新验证，不能只重打 ZIP。
