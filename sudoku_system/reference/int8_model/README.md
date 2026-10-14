# reference/int8_model/ — 软件 INT8 参考模型（不在上板流程里）

队员 dyx 交付的原版识别程序，原来在 `dyx_camera/processing/`。保留在这里只用来**对照**：
上板识别用的是 `cnn_solver.bit` 里的硬件 CNN，不是这个软件模型。

| 文件 | 作用 |
|---|---|
| `recognize.py` | 原版：整张照片 → 切 81 格 → 软件 INT8 识别，输出一整套检查材料 |
| `model/predict.py` | 浮点 / INT8 推理与十类自检（浮点部分需要 PyTorch） |
| `model/preprocess.py` | 和 `sudoku/preprocess.py` 内容完全相同 |
| `model/weights/int8.npz`、`int8.json` | INT8 权重和量化参数（`board_verified: false`） |
| `model/self_test.npz` | 十类自检样本 |

`sudoku/grid.py` 就是从 `recognize.py` 的切格部分整理出来的，切格结果逐像素一致。

## 用法（电脑或板子上都可以，不需要 PyTorch）

```bash
cd reference/int8_model
python recognize.py ../../tests/data/sample_capture.png --output ../../results/ref_check
```

输出目录里有识别出的盘面 `predicted_board.txt`、各阶段中间图和 `result.json`。
可以和硬件 CNN 的识别结果对比，找出两边不一致的格子。
