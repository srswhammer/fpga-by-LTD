"""板上使用：预处理单格图片，再将784个像素写入BRAM。"""
import numpy as np

from preprocess import preprocess_sudoku_cell


# 从已加载Overlay的HWH信息取得BRAM地址；ip_name填写实际AXI BRAM控制器名称。
def open_bram(overlay, ip_name):
    from pynq import MMIO

    info = overlay.ip_dict[ip_name]
    return MMIO(info['phys_addr'], info['addr_range'])


# 输入约定为uint8[28,28]，按行排列；每个32位字的低8位存较前的像素。
def write_pixels(bram, pixels):
    words = np.frombuffer(pixels.tobytes(order='C'), dtype='<u4')
    for index, word in enumerate(words):
        bram.write(index * 4, int(word))
    bram.read(780)  # 回读末字，使写入完成后再交给外部程序启动CNN。


# 原始单格图片先调用原有预处理；返回最终像素，便于查看。
def write_image(bram, image, polarity='auto'):
    pixels = preprocess_sudoku_cell(image, polarity=polarity).normalized_uint8
    write_pixels(bram, pixels)
    return pixels
