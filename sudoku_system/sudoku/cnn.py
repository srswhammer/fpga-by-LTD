"""板上使用：逐格写 BRAM -> 启动 CNN -> 等完成。

来源：cnn_sudoku/cnn_board.py（组长）。唯一改动：recognize_images() 里的导入
改成从同一个包里的 preprocess.py 导入（放进 sudoku 包后的写法）。

Block Design（bd/build_bd.tcl 搭的）里：
  cnn_bram：AXI BRAM Controller，PS 往这里写一格 784 像素（地址 0x40000000）
  cnn_gpio：AXI GPIO，通道 1 输出 start，通道 2 输入 status（地址 0x41200000）
status = {cell_count[6:0], last_digit[3:0], done}
"""
import time

import numpy as np
from pynq import MMIO

GPIO_DATA = 0x0     # 通道 1：start
GPIO2_DATA = 0x8    # 通道 2：status


def open_mmio(overlay, name, default_addr, default_range):
    """按 IP 名字找地址。BRAM 控制器在 PYNQ 里放在 mem_dict，不在 ip_dict。"""
    for d in (overlay.ip_dict, overlay.mem_dict):
        if name in d:
            info = d[name]
            return MMIO(info['phys_addr'], info['addr_range'])
    return MMIO(default_addr, default_range)


def open_cnn(overlay):
    bram = open_mmio(overlay, 'cnn_bram', 0x40000000, 0x1000)
    gpio = open_mmio(overlay, 'cnn_gpio', 0x41200000, 0x10000)
    gpio.write(GPIO_DATA, 0)
    return bram, gpio


def write_pixels(bram, pixels):
    """pixels：uint8[28,28]（或 784 个），按行排列；每个 32 位字的低 8 位放靠前的像素。"""
    pixels = np.asarray(pixels, dtype=np.uint8).reshape(784)
    words = np.frombuffer(pixels.tobytes(), dtype='<u4')
    for i, w in enumerate(words):
        bram.write(i * 4, int(w))
    bram.read(780)  # 回读一次，确保写完


def run_cell(gpio, timeout=0.1):
    """BRAM 已写好一格后调用：启动 CNN，等 784 像素读完。"""
    gpio.write(GPIO_DATA, 1)
    t0 = time.time()
    while not (gpio.read(GPIO2_DATA) & 1):
        if time.time() - t0 > timeout:
            gpio.write(GPIO_DATA, 0)
            raise TimeoutError('CNN 没有读走这一格，检查求解器是否一直不收（cell_ready=0）')
    gpio.write(GPIO_DATA, 0)


def last_digit(gpio):
    """最近一格送给求解器的数字（0 = 空格）。"""
    return (gpio.read(GPIO2_DATA) >> 1) & 0xF


def recognize(bram, gpio, pixels):
    """识别一格（cell_ready 接常 1 时用）：写图、启动、等算完，返回数字。"""
    write_pixels(bram, pixels)
    run_cell(gpio)
    time.sleep(0.0005)          # 一格约 0.1 ms 算完
    return last_digit(gpio)


def recognize_images(overlay, images, polarity='dark_on_light'):
    """images：81 张单格原始图片，用队友的 preprocess.py 预处理后识别。"""
    from .preprocess import preprocess_sudoku_cell
    bram, gpio = open_cnn(overlay)
    return [recognize(bram, gpio, preprocess_sudoku_cell(img, polarity=polarity).normalized_uint8)
            for img in images]
