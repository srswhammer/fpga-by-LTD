# 图像处理交接

`cells/00.png`～`cells/80.png` 是裁好、去格线并做光照修正的单通道灰度图片，黑字浅底，空白格保留。编号按每行从左到右、各行从上到下：`编号 = 行索引*9 + 列索引`，索引从0开始。

这些图片尚未缩放居中到28×28，也未反色；交给TBR的 `preprocess_sudoku_cell(..., polarity='dark_on_light')`，随后由组长的程序接CNN和求解。不要将PNG压缩字节直接当作像素写BRAM。

本次提交是 `sudoku_test_1008_01.png` 实拍样本的81格，已在电脑端核对经TBR预处理后与原模型输入逐像素一致。后续可用实际拍摄照片更新：

```powershell
python C_系统联调/sudoku_photo/recognize.py "照片.png" --output "新的检查目录" --cells-dir pynq/cells --cells-only
```

`--cells-only` 不运行软件CNN预测；默认完整识别也会在结果目录生成同样命名的 `cells/`。显式指定交接目录会更新其中的00～80图片，生成失败时不要交付未完整的目录。
