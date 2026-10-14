"""数独识别求解系统：拍照 → 切格 → 硬件 CNN 识别 → 硬件求解 → LCD 显示。

模块：
  camera      拍照（camera.bit）
  grid        找棋盘、切 81 格
  preprocess  单格预处理成 28×28
  cnn         写 BRAM、启动 CNN（cnn_solver.bit）
  solver      81 格送入 CNN + 求解器，读回答案
  lcd         校验答案、打包 81 字节、经电脑桥发送（lcd.bit）
  pipeline    把以上串成一次运行：from sudoku.pipeline import run; run()

这里故意不自动导入子模块：在电脑上也能单独 import grid / lcd 做测试，
不会因为没有 pynq 而报错。
"""
