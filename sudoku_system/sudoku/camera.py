"""摄像头拍照：加载 camera.bit，通过 VDMA 抓一帧 640×480 RGB565，转成 RGB 图像。

来源：dyx_one_click/camera_runtime.py（和 dyx_camera/camera_export.ipynb 里的拍照代码相同）
改动：
  - 原来一 import 就加载 bit、分配内存；现在包成 Camera 类，调用 open() 才加载
  - bit 路径固定为 overlays/camera.bit，不再依赖"当前目录"
  - 原 run_all.py 在 finally 里做的收尾（停 VDMA、释放缓冲区）放进 close()
  - VDMA 寄存器的写入顺序、数值、等待时间都和原来一样
  - bit 路径和取帧等待时间在 config.py；640×480 由 bit 决定，留在本文件

用法：
    with Camera() as cam:          # 进入时加载 bit，退出时自动收尾（出错也会收尾）
        photo = cam.take_photo('capture.png', show=True)
"""
import time
from pathlib import Path

import numpy as np

from . import config

# 由 camera.bit 里的摄像头前端决定，不是可调参数
WIDTH = 640
HEIGHT = 480
BYTES_PER_PIXEL = 2

# AXI VDMA 写通道（S2MM）寄存器偏移
S2MM_CR = 0x30       # 控制：0x04 复位，0x03 启动，0x00 停止
S2MM_SR = 0x34       # 状态
S2MM_VSIZE = 0xA0    # 行数（写它会启动传输，所以最后写）
S2MM_HSIZE = 0xA4    # 每行字节数
S2MM_STRIDE = 0xA8   # 行跨度
S2MM_ADDR = 0xAC     # 帧缓冲物理地址


def rgb565_to_rgb(raw):
    """uint16 RGB565 → uint8 RGB（和原代码的换算完全一样）。"""
    r = (((raw >> 11) & 0x1F) * 255 // 31).astype(np.uint8)
    g = (((raw >> 5) & 0x3F) * 255 // 63).astype(np.uint8)
    b = ((raw & 0x1F) * 255 // 31).astype(np.uint8)
    return np.dstack((r, g, b))


class Camera:
    def __init__(self, bitfile=None, width=WIDTH, height=HEIGHT, verbose=True):
        self.bitfile = Path(bitfile or config.CAMERA_BIT)
        self.width = width
        self.height = height
        self.verbose = verbose
        self.overlay = None
        self.frame = None
        self.vdma = None

    def open(self):
        from pynq import Overlay, allocate, MMIO     # 只在板子上需要
        self.overlay = Overlay(str(self.bitfile), download=True)
        if self.verbose:
            print('Overlay加载成功：', self.overlay.is_loaded())
            print('可用IP：', list(self.overlay.ip_dict.keys()))
        self.frame = allocate(shape=(self.height, self.width), dtype=np.uint16)
        info = self.overlay.ip_dict['vdma']
        self.vdma = MMIO(info['phys_addr'], info['addr_range'])
        if self.verbose:
            print('缓冲区地址：', hex(self.frame.physical_address))
        return self

    def take_photo(self, save_path=None, show=False):
        """抓一帧，返回 (480, 640, 3) 的 RGB uint8 数组。"""
        if self.vdma is None:
            raise RuntimeError('请先调用 open()，或用 with Camera() as cam: 的写法')
        frame, vdma = self.frame, self.vdma
        frame[:] = 0
        frame.flush()

        vdma.write(S2MM_CR, 0x04)
        time.sleep(0.2)
        vdma.write(S2MM_CR, 0x03)
        vdma.write(S2MM_ADDR, frame.physical_address)
        vdma.write(S2MM_STRIDE, self.width * BYTES_PER_PIXEL)
        vdma.write(S2MM_HSIZE, self.width * BYTES_PER_PIXEL)
        vdma.write(S2MM_VSIZE, self.height)

        time.sleep(config.CAPTURE_WAIT)
        status = vdma.read(S2MM_SR)
        vdma.write(S2MM_CR, 0x00)
        time.sleep(0.2)
        frame.invalidate()

        image = rgb565_to_rgb(np.array(frame, copy=True))

        if save_path:
            from PIL import Image
            Image.fromarray(image).save(str(save_path))
            print('已保存：', save_path)
        if show:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 7))
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        if self.verbose:
            print('VDMA状态：', hex(status))
        return image

    def close(self):
        """停 VDMA、释放帧缓冲。可以重复调用。"""
        if self.vdma is not None:
            self.vdma.write(S2MM_CR, 0)
            self.vdma = None
        if self.frame is not None:
            self.frame.freebuffer()
            self.frame = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False
