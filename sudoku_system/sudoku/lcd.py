"""LCD 显示：校验答案 → 打包 81 字节 → 加载 LCD bit → 经电脑桥发送并等待 ACK。

来源：dyx_one_click/run_all.py
  - request_json()     原样搬过来
  - validate_result()  原样搬过来
  - check_bridge()     来自 main() 开头检查电脑桥的那两行
  - save_result()      来自 main() 里写 lcd_result.json / .bin 的几行
  - show()             来自 main() 的第 3/3 步
新增：load_payload()，给分步调试读回上一步保存的结果。
地址、bit 路径、超时等设置都在 config.py。
"""
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from . import config


def request_json(url, data=None, timeout=None):
    """data 为 None 时发 GET，否则把 data 转成 JSON 用 POST 发出；返回对方回的 JSON。"""
    timeout = config.BRIDGE_TIMEOUT if timeout is None else timeout
    payload = None if data is None else json.dumps(data).encode('utf-8')
    req = Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def validate_result(status, puzzle, answer, given):
    """检查求解结果是否合法，合法就打包成 LCD 用的 81 个字节。

    每字节：低 4 位 = 数字；bit4 = 0 原题 / 1 补入。
    """
    if status != 0:
        raise RuntimeError('Solver status %s; LCD will not be loaded' % status)
    puzzle = np.asarray(puzzle)
    answer = np.asarray(answer)
    given = np.asarray(given, dtype=bool)
    if puzzle.shape != (9, 9) or answer.shape != (9, 9) or given.shape != (9, 9):
        raise ValueError('Expected 9x9 result')
    if not np.array_equal(given, puzzle != 0) or not np.array_equal(answer[given], puzzle[given]):
        raise ValueError('Original digits do not match solution')
    units = list(answer) + list(answer.T)
    units += [answer[r:r+3, c:c+3].ravel() for r in (0, 3, 6) for c in (0, 3, 6)]
    if any(sorted(unit.tolist()) != list(range(1, 10)) for unit in units):
        raise ValueError('Invalid Sudoku solution')
    payload = answer.astype(np.uint8).copy()
    payload[~given] |= 0x10
    return payload.ravel().tolist()


def check_bridge(url=None):
    """确认电脑上的 lcd_bridge.py 已经启动；没启动就直接报错，不往下跑。"""
    url = url or config.BRIDGE_URL
    if not request_json(url + '/health').get('ready'):
        raise RuntimeError('PC LCD bridge is not ready')


def save_result(run_dir, puzzle, answer, payload, steps, source=''):
    """把本次结果存成 lcd_result.json 和 lcd_result.bin（格式和原来一样）。"""
    run_dir = Path(run_dir)
    result = dict(original=np.asarray(puzzle).tolist(), solution=np.asarray(answer).tolist(),
                  display_bytes=payload, steps=int(steps), source=str(source))
    (run_dir / 'lcd_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                             encoding='utf-8')
    (run_dir / 'lcd_result.bin').write_bytes(bytes(payload))
    return result


def load_payload(path):
    """从 lcd_result.json 读回 81 个显示字节。"""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))['display_bytes']
    if len(payload) != 81:
        raise ValueError('Expected 81 display bytes')
    return payload


def show(payload, bitfile=None, url=None, settle=None):
    """加载 LCD bit，通过电脑桥发送 81 字节，等待 ACK。不传的参数用 config 里的值。"""
    from pynq import Bitstream          # 放在函数里：在电脑上 import 本文件也不会报错
    bitfile = bitfile or config.LCD_BIT
    url = url or config.BRIDGE_URL
    settle = config.LCD_SETTLE if settle is None else settle
    Bitstream(str(bitfile)).download()
    time.sleep(settle)
    reply = request_json(url + '/display', {'display_bytes': list(payload)})
    if reply.get('ack') != config.LCD_ACK:
        raise RuntimeError('LCD did not acknowledge')
    return reply
