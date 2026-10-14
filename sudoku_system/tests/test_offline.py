"""离线自检：不加载任何 bit、不需要摄像头和电脑桥，检查软件部分有没有被改坏。

检查项：
  1. 三个 bit 和 hwh 都在 overlays/ 里
  2. 切格：用样例照片 data/sample_capture.png 重新切 81 格，和当时板子上切出的
     data/sample_cells/ 对比。板子上必须逐像素一致；电脑上 OpenCV 版本不同，
     只要求每格位置和尺寸一致（灰度会差几个级）
  3. 预处理：81 格都能变成 28×28 uint8
  4. LCD 打包：用 data/sample_lcd_result.json 重新打包 81 字节，应与文件里一致；
     无解、答案不合法时必须报错

运行：板子或电脑上都可以
      python3 tests/test_offline.py        或在 Jupyter 里 %run test_offline.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sudoku import config, grid, lcd
from sudoku.preprocess import preprocess_sudoku_cell

DATA = HERE / 'data'
results = []


def check(name, ok, detail=''):
    results.append(ok)
    print('%s  %s%s' % ('PASS' if ok else 'FAIL', name, ('：' + detail) if detail else ''))


# 1. 文件齐全
missing = [p.name for p in config.REQUIRED_FILES if not p.is_file()]
check('overlays 文件齐全', not missing, '缺少 ' + ', '.join(missing) if missing else '')

# 2. 切格
cells = grid.split(DATA / 'sample_capture.png')
expected = grid.load_cells(DATA / 'sample_cells')
same_shape = len(cells) == 81 and all(a.shape == b.shape for a, b in zip(cells, expected))
max_diff = max(int(np.abs(a.astype(int) - b.astype(int)).max()) for a, b in zip(cells, expected)) \
    if same_shape else -1
try:
    import pynq  # noqa: F401  能导入说明在板子上
    on_board = True
except ImportError:
    on_board = False
if not same_shape:
    check('切格位置和尺寸与样例一致', False, '有格子尺寸不同，找棋盘或切格结果变了')
elif max_diff == 0:
    check('切格与板上样例逐像素一致', True)
elif on_board:
    check('切格与板上样例逐像素一致', False, '灰度最大差 %d；板子上应完全一致' % max_diff)
else:
    check('切格位置和尺寸与样例一致', True,
          '不在板子上：OpenCV 版本不同，灰度最大差 %d，属正常；板子上应为 0' % max_diff)

# 3. 预处理
inputs = np.asarray([preprocess_sudoku_cell(c, polarity=config.CELL_POLARITY).normalized_uint8
                     for c in cells], dtype=np.uint8)
check('预处理输出 (81, 28, 28) uint8', inputs.shape == (81, 28, 28), str(inputs.shape))

# 4. LCD 打包
ref = json.loads((DATA / 'sample_lcd_result.json').read_text(encoding='utf-8'))
puzzle, answer = np.array(ref['original']), np.array(ref['solution'])
payload = lcd.validate_result(0, puzzle, answer, puzzle != 0)
check('LCD 81 字节与样例一致', payload == ref['display_bytes'])

for label, args in [('无解状态被拦截', (2, puzzle, answer, puzzle != 0)),
                    ('非法答案被拦截', (0, puzzle, np.roll(answer, 1, axis=1), puzzle != 0))]:
    try:
        lcd.validate_result(*args)
        check(label, False, '没有报错')
    except (RuntimeError, ValueError):
        check(label, True)

print('\n全部通过' if all(results) else '\n有 %d 项未通过' % results.count(False))
