# 数独拍照与识别

指定照片完成棋盘定位、透视校正、光照修正、81格切分及陶博然v2 INT8识别。已接通开发板上的实时拍照与ARM软件识别，求解和LCD全链路尚未接入。

## 分工

丁昀熙负责采集链路、棋盘定位校正、切格去线、光照修正及结果编码。单格缩放居中和28×28白字黑底UINT8输出复用陶博然的预处理函数，数字判断调用其INT8模型。算法中不使用已知题目或答案更改预测。

## 在开发板上使用

已验证环境：Python 3.8.2、NumPy 1.20.3、OpenCV 4.2.0、Pillow 7.0.0、PYNQ 2.7.0。

1. 电脑端运行 `python build_board_bundle.py --name sudoku_board_review`，生成 `results/sudoku_board_review.zip`。
2. 在板上 `camera_capture` 目录上传并解压该包，上传本目录的 `camera_recognize.ipynb`；同目录还需配套 `camera_capture.bit` 和 `camera_capture.hwh`。
3. 关闭原有拍照Notebook内核后，打开新版并依次运行四个代码格。初始化后，只重复最后一格拍照识别。
4. 结果写入 `sudoku_board_review/results/live_*/`。核对原图及输出数字后再用于下游模块。

上传包对模型预处理的类型声明和Pillow枚举做旧版本兼容，权重保持不变；INT8分批处理以降低内存需求。直接调用团队原有整数推理函数，运行前进行随包自检。

整理版已清除现场测试格和图片输出，独立加载最新程序，不依赖 `thin`、`grid`、`review` 等临时变量；通过Python 3.8语法检查，最终整理后的版本尚待重新从头上板验证。采集仍沿用已有固定等待及VDMA状态输出，帧同步状态的进一步排查待完成。

## 在电脑上使用

Python 3.10或更高版本，在本目录执行：

```powershell
python -m pip install -r requirements.txt
python recognize.py "C:\Users\dingy\Desktop\sudoku_test_1008_01.png"
```

默认读取仓库 `B_模型交接_v2`，输出到 `results/照片文件名/`。命令行不覆盖非空结果目录，重复运行请用 `--output results/check_again` 指定新目录。`--model` 只应指定可信团队模型目录；模型更新后需重新验证。

## 输出与接口

组长约定的图片交接目录为仓库根目录 `pynq/cells/`，文件名 `00.png`～`80.png`，包含空白格，按行排列。使用 `--cells-dir pynq/cells --cells-only` 导出裁切、去线及光照修正后的黑字浅底灰度单格，后续由组长调用TBR单格预处理及硬件CNN。默认完整流程在结果目录的 `cells/` 导出同样的图片。目录说明及当前实拍交接样本见 `pynq/README.md`。

- `predicted_board.txt`：九行九列预测数字，0为空白。
- `01_detected_board.png`、`02_rectified_board.png`：棋盘定位及校正图。
- `03_raw_cells_contact.png`、`04_model_inputs_contact.png`：81格原图与主路模型输入；单格图片保存在 `cells_raw/`、`cells_28x28/`。
- `model_inputs_uint8.npy` / `.bin`：81×28×28 UINT8输入，二进制63504字节。
- `result.json`：数字、独热码、位置及预处理记录，也包含 `review_cells`、`requires_review`；`scores_int32.npy` 是主路模型分数，不是概率。
- `comparison_inputs_uint8.npy`、`05_comparison_inputs_contact.png`：无中值滤波对照路输入。
- `one_hot_81.txt`：9位独热码，1对应最低位，空白全零。
- `original_81bytes.bin` / `.hex`：候选原题，一格一字节，共81字节；bit7–5=0，bit4=0代表原题，bit3–0为0–9的8421编码。求解补入数字的bit4应为1，当前不生成求解结果。

顺序为每行从左到右、各行从上到下。81字节文件没有通信帧头、校验或握手，不自动发往LCD。正式传输仍需对接接收模块。

## 验证与局限（2026-10-09）

- 三张历史照片已在板上验证一致，旧版识别耗时7.04～7.92秒；粗笔正面、倾斜及细笔实时照片均曾通过，旧版拍照至识别约13秒。
- 细笔样本出现过2→3和7→1；主要连通笔画筛选修复了已知失败样本，并通过板上三张历史照片回归。
- 外框裁切导致的定位失败已用保留余量的格线检查修复，失败照片在板上通过；仍保留主线验证。
- 后续短横画7仍可能认成1。双路对照会提示预测分歧，但板上已观察到漏报；也会将正确的空白格标为可疑。`requires_review=false` 不代表结果正确。当前81字节输出需人工核对，不能靠该标志自动放行给求解模块。
- 调整字迹后的现场样本主路81格正确，双路对照将第8行第3列空白提示为0/2，耗时16.26秒。最新现场验证样本为 `live_c48e7446`。

以上为同一道题、有限字迹和拍摄条件的开发验证，不能代表总体准确率。识别目前在ARM上运行；筛选假设笔画主体连通，断笔需更多样本验证。保持四角完整入镜、数字朝上，当前不支持任意旋转方向纠正。

`results/` 已忽略，照片及运行结果不会自动进入Git提交。
