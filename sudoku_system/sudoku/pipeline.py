"""主流程：拍照 → 切格 → 硬件 CNN 识别 + 求解 → LCD 显示。

来源：dyx_one_click/run_all.py 的 main()。流程、顺序、检查项都和原来一样，区别是：
  - 不再需要 os.chdir / sys.path.insert / sys.modules.pop / exec，改成普通的包内导入
  - bit 路径都从 overlays/ 拼出来，从哪个目录运行都一样
  - 81 格在内存里直接传给预处理，不必先存 PNG 再读回（结果相同，PNG 仍会存一份供检查）
  - 每一步单独成函数，分步调试的 Notebook 可以只调用其中一步

仍然是依次加载三个完整 bit（摄像头 → CNN/求解器 → LCD），LCD 仍经电脑桥发送。
路径、开关、超时都在 config.py。

用法：
    from sudoku.pipeline import run
    result = run()
"""
import time
import uuid
from pathlib import Path

import numpy as np

from . import config, grid, lcd
from .camera import Camera
from .preprocess import preprocess_sudoku_cell
from .solver import solve_cells, print_board, STATUS_TEXT


def check_files():
    """开跑前确认三个 bit（和对应的 hwh）都在。"""
    missing = [str(p) for p in config.REQUIRED_FILES if not p.is_file()]
    if missing:
        raise FileNotFoundError('缺少文件：\n  ' + '\n  '.join(missing))


def new_run_dir(prefix=None):
    """每次运行一个新目录：results/live_xxxxxxxx/"""
    prefix = prefix or config.RUN_PREFIX
    run_dir = config.RESULTS / ('%s_%s' % (prefix, uuid.uuid4().hex[:8]))
    run_dir.mkdir(parents=True)
    return run_dir


def capture(run_dir=None, show=None):
    """第 1 步前半：加载 camera.bit 拍一张照片，返回 RGB 数组。"""
    show = config.SHOW_PHOTO if show is None else show
    save_path = run_dir / 'capture.png' if run_dir else None
    with Camera() as cam:
        return cam.take_photo(save_path, show=show)


def split_cells(photo, run_dir=None):
    """第 1 步后半：切出 81 格。给了 run_dir 就保存中间图和 cells/00～80.png。"""
    cells = grid.split(photo,
                       save_dir=run_dir / 'processing' if run_dir else None,
                       cells_dir=run_dir / 'cells' if run_dir else None)
    if len(cells) != 81:
        raise RuntimeError('81 fresh cells were not exported')
    return cells


def solve(cells):
    """第 2 步：81 格预处理成 28×28 → 加载 cnn_solver.bit → 硬件识别 + 求解。"""
    from pynq import Overlay                     # 只在板子上需要
    inputs = np.asarray([preprocess_sudoku_cell(c, polarity=config.CELL_POLARITY).normalized_uint8
                         for c in cells], dtype=np.uint8)
    if inputs.shape != (81, 28, 28):
        raise ValueError('Invalid CNN input shape')
    overlay = Overlay(str(config.CNN_BIT))
    status, puzzle, answer, given, steps = solve_cells(overlay, inputs, timeout=config.SOLVER_TIMEOUT)
    print(STATUS_TEXT.get(status, str(status)))
    print(puzzle)
    print_board(answer, given)
    return dict(status=status, puzzle=puzzle, answer=answer, given=given, steps=steps)


def display(result, run_dir=None, source=''):
    """第 3 步：校验答案 → 保存 lcd_result → 加载 lcd.bit → 经电脑桥发送并等 ACK。"""
    payload = lcd.validate_result(result['status'], result['puzzle'],
                                  result['answer'], result['given'])
    if run_dir:
        lcd.save_result(run_dir, result['puzzle'], result['answer'], payload,
                        result['steps'], source)
    lcd.show(payload)
    return payload


def run(show_photo=None, save_debug=None):
    """一次完整运行。任何一步失败都会直接报错停下，不会拿旧图片顶替。

    show_photo / save_debug 不传就用 config 里的 SHOW_PHOTO / SAVE_DEBUG。
    """
    save_debug = config.SAVE_DEBUG if save_debug is None else save_debug
    check_files()
    lcd.check_bridge()                           # 电脑桥没开就别开始拍照
    run_dir = new_run_dir()
    started = time.perf_counter()

    print('1/3 拍照和切格')
    photo = capture(run_dir, show=show_photo)
    cells = split_cells(photo, run_dir if save_debug else None)

    print('2/3 硬件 CNN 识别和求解')
    result = solve(cells)

    print('3/3 切换 LCD 并发送本次答案')
    result['payload'] = display(result, run_dir, source=run_dir / 'capture.png')

    print('完成：LCD已确认接收本次答案；总耗时 %.2f 秒' % (time.perf_counter() - started))
    print('本次记录：', run_dir)
    result['run_dir'] = run_dir
    return result
