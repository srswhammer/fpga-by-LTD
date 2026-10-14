"""板上使用（第 2 步整合版）：81 格图片 → CNN → 求解器 → 读答案。

来源：cnn_sudoku/sudoku_board.py（组长）。改动只有两处：
  1. 导入改成从同一个包里的 cnn.py 导入（文件改名为 cnn.py，并放进 sudoku 包）
  2. 求解器默认地址 0x43C00000 → 0x40002000（hwh 里的实际地址；平时按名字 'sudoku' 查到的就是它）

Block Design（bd/build_bd_full.tcl 搭的）里：
  cnn_bram：AXI BRAM Controller，PS 往这里写一格 784 像素
  cnn_gpio：AXI GPIO，通道 1 输出 start，通道 2 输入 status
  sudoku  ：求解器 sudoku_axi_cnn（AXI-Lite），CNN 结果在 PL 里直接送进来
依赖 cnn.py（写 BRAM、启动一格）。
"""
import time

import numpy as np

from .cnn import open_cnn, open_mmio, write_pixels, run_cell

CTRL, STATUS, CYCLES, FRAME = 0x00, 0x04, 0x08, 0x18
IN0, OUT0, GIVEN0 = 0x40, 0x80, 0xC0
STATUS_TEXT = {0: '解出', 1: '原题矛盾', 2: '无解', 3: '栈溢出'}


def open_sudoku(overlay):
    return open_mmio(overlay, 'sudoku', 0x40002000, 0x1000)


def read_cells(mm, base):
    """读 11 个字，拆成 81 个 4 位数字（第 8w+n 格在字 w 的 bit[4n+3:4n]）。"""
    out = []
    for w in range(11):
        v = mm.read(base + 4 * w)
        out += [(v >> (4 * n)) & 0xF for n in range(8)]
    return np.array(out[:81]).reshape(9, 9)


def read_given(mm):
    bits = 0
    for w in range(3):
        bits |= mm.read(GIVEN0 + 4 * w) << (32 * w)
    return np.array([(bits >> i) & 1 for i in range(81)], dtype=bool).reshape(9, 9)


def solve_cells(overlay, cells, timeout=1.0):
    """cells：81 张已预处理好的 28x28 uint8 图片（白字黑底），从左到右、从上到下。

    返回 (status, 识别出的题目 9x9, 答案 9x9, 原题掩码 9x9, 求解周期数)
    """
    bram, gpio = open_cnn(overlay)
    sd = open_sudoku(overlay)

    frame0 = sd.read(FRAME) & 0xFFFF
    for img in cells:                       # 81 格逐个送进 CNN
        write_pixels(bram, img)
        run_cell(gpio)

    t0 = time.time()                        # 等最后一格算完、求解器跑完
    while (sd.read(FRAME) & 0xFFFF) == frame0 or not (sd.read(CTRL) & 2):
        if time.time() - t0 > timeout:
            raise TimeoutError('求解器没有被 CNN 启动，检查 cell_* 连线')
    if sd.read(FRAME) >> 31:
        print('警告：frame_err = 1，CNN 送格数和 cell_last 对不上')

    status = sd.read(STATUS)
    given = read_given(sd)
    answer = read_cells(sd, OUT0)
    puzzle = np.where(given, answer, 0)     # 原题 = 答案里原题那些格
    return status, puzzle, answer, given, sd.read(CYCLES)


def print_board(answer, given):
    """原题数字原样显示，求解填的数字用红色显示。"""
    for r in range(9):
        if r % 3 == 0:
            print('+-------+-------+-------+')
        row = ''
        for c in range(9):
            if c % 3 == 0:
                row += '| '
            d = answer[r, c]
            row += (f'{d} ' if given[r, c] else f'\033[31m{d}\033[0m ')
        print(row + '|')
    print('+-------+-------+-------+')
