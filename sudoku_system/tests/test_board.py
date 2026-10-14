"""上板自检：用仿真里同一组 81 格图片（data/joint_cells.mem）跑硬件 CNN + 求解器，结果应与仿真一致。

来源：cnn_sudoku/test_board.py（组长）。只改了导入方式和文件路径，判断逻辑不变。
不需要摄像头、LCD 和电脑桥；用来确认 cnn_solver.bit 和 sudoku 包本身没问题。

运行：在 Jupyter 里打开 tests 目录，新建 Notebook 执行 %run test_board.py
      或在终端：sudo python3 tests/test_board.py
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # 找到 sudoku 包

from pynq import Overlay

from sudoku import config
from sudoku.solver import solve_cells, print_board, STATUS_TEXT

DATA = HERE / 'data'

ol = Overlay(str(config.CNN_BIT))
print('IP 列表：', list(ol.ip_dict.keys()), list(ol.mem_dict.keys()))

pix = np.array([int(x, 16) for x in (DATA / 'joint_cells.mem').read_text().split()], dtype=np.uint8)
cells = pix.reshape(81, 28, 28)
exp_puz = np.array([int(x) for x in (DATA / 'joint_puzzle.txt').read_text().split()]).reshape(9, 9)
exp_sol = np.array([int(x) for x in (DATA / 'joint_solution.txt').read_text().split()]).reshape(9, 9)

t0 = time.time()
status, puzzle, answer, given, steps = solve_cells(ol, cells)
t1 = time.time()

print('STATUS =', status, STATUS_TEXT.get(status, '?'))
print('求解步数 =', steps, '（实际时钟周期 = 步数 x 4）')
print('PS 送图 + 识别 + 求解 总用时 %.1f ms' % ((t1 - t0) * 1000))
print_board(answer, given)

ok_puz = (puzzle == exp_puz).all()
ok_sol = (answer == exp_sol).all()
print('识别结果', '正确' if ok_puz else '有错', '；答案', '正确' if ok_sol else '有错')
if not ok_puz:
    for r, c in zip(*np.nonzero(puzzle != exp_puz)):
        print('  第 %d 行第 %d 列：识别 %d，应为 %d' % (r + 1, c + 1, puzzle[r, c], exp_puz[r, c]))
print('PASS' if ok_puz and ok_sol and status == 0 else 'FAIL')
