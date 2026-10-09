"""板端入口：Python 预处理 → PYNQ MMIO 写 BRAM → FPGA 预测 → 9 位编码。

需配套实现《板上BRAM接口.md》的硬件；不是 CPU 推理，也不自动下载 bitstream。
"""
import argparse
import math
import sys
import threading
import time

import numpy as np

from preprocess import preprocess_sudoku_cell


# 以下均为同一块 4 KiB BRAM 内的字节偏移；物理基地址从 HWH 获取。
MAGIC, VERSION, MODEL, STATUS = 0x00, 0x04, 0x08, 0x0C
REQUEST, ACK, RESULT, ERROR = 0x10, 0x14, 0x18, 0x1C
PIXELS, BRAM_BYTES = 0x100, 4096
MAGIC_VALUE, VERSION_VALUE, MODEL_VALUE = 0x434E4E32, 1, 0xEE067A81
READY, BUSY, FAULT = 1, 2, 4


class BramPredictor:
    """复用一个实例逐格识别；同一硬件只允许一个进程使用。"""

    def __init__(self, mmio, timeout=2.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout 必须是有限正数，单位为秒')
        self.mmio = mmio
        self.timeout = timeout
        self.lock = threading.Lock()
        self.failed = False
        self._check_identity()

    @classmethod
    def from_overlay(cls, overlay, ip_name='cnn_bram', timeout=2.0):
        # 复用摄像头所在的完整 Overlay，避免重新配置 FPGA 导致相机中断。
        from pynq import MMIO

        if not overlay.is_loaded():
            raise RuntimeError('请先加载包含相机和 CNN 的匹配 Overlay')
        info = overlay.ip_dict[ip_name]
        if 'axi_bram_ctrl' not in info['type'] or info['addr_range'] < BRAM_BYTES:
            raise ValueError('所选 IP 必须是映射至少 4 KiB 的 AXI BRAM Controller')
        mmio = MMIO(info['phys_addr'], BRAM_BYTES)
        return cls(mmio, timeout)

    def _check_identity(self):
        actual = tuple(int(self.mmio.read(offset)) for offset in (MAGIC, VERSION, MODEL))
        if actual != (MAGIC_VALUE, VERSION_VALUE, MODEL_VALUE):
            raise RuntimeError('BRAM 协议或 v2.4 模型标识不匹配；不能使用旧 CNN 或相机专用 Overlay')

    def predict_uint8(self, pixels):
        """接收已预处理的 uint8[28,28]；不再缩放、归一化或保存图片文件。"""
        pixels = np.asarray(pixels)
        if pixels.shape != (28, 28) or pixels.dtype != np.uint8:
            raise ValueError('FPGA 输入必须为 uint8 类型的 28×28 单通道数组')
        words = np.frombuffer(pixels.tobytes(order='C'), dtype='<u4')

        with self.lock:
            if self.failed:
                raise RuntimeError('上次硬件调用失败，请检查并复位硬件后重新创建实例')
            self._check_identity()
            status = int(self.mmio.read(STATUS))
            request, ack = int(self.mmio.read(REQUEST)), int(self.mmio.read(ACK))
            if status != READY or request != ack:
                raise RuntimeError('CNN 尚未就绪或仍有未完成请求，禁止覆盖输入 BRAM')

            # 每个32位字存4个连续像素：第一个像素在低8位。先完整写图，再提交序号。
            try:
                for index, word in enumerate(words):
                    self.mmio.write(PIXELS + index * 4, int(word))
                if int(self.mmio.read(PIXELS + 780)) != int(words[-1]):
                    raise RuntimeError('BRAM 末字回读不一致，未启动 CNN')
                sequence = (ack + 1) & 0xFFFFFFFF
                self.mmio.write(REQUEST, sequence)

                # ACK 是持续保存的完成序号，不是单周期 done 脉冲，Python 不会漏采。
                deadline = time.monotonic() + self.timeout
                while int(self.mmio.read(ACK)) != sequence:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('等待 FPGA 超时；保留现场，不自动重发或覆盖图片')
                    time.sleep(0.001)
                self._check_identity()
                status, error = int(self.mmio.read(STATUS)), int(self.mmio.read(ERROR))
                if status != READY or error:
                    raise RuntimeError(f'FPGA 计算失败：status=0x{status:X}, error=0x{error:X}')
                code = int(self.mmio.read(RESULT))
                if code > 0x1FF or code < 0 or (code & (code - 1)):
                    raise RuntimeError('FPGA 返回的结果不是合法9位独热编码')
                return f'{code:09b}'
            except Exception:
                self.failed = True
                raise

    def predict_image(self, image, polarity='auto'):
        """接收已切分的单格路径、PIL 图片或数组；与现有模型使用相同预处理。"""
        prepared = preprocess_sudoku_cell(image, polarity=polarity)
        if 'warning' in prepared.metadata:
            print(prepared.metadata['warning'], file=sys.stderr)
        return self.predict_uint8(prepared.normalized_uint8)


# 板上命令入口只绑定已经加载的完整 Overlay；不会悄悄退回 CPU CNN。
def main():
    parser = argparse.ArgumentParser(description='板上 Python 预处理及 BRAM 硬件预测')
    parser.add_argument('--overlay', required=True, help='已加载的完整 .bit 路径，旁边需有同名 .hwh')
    parser.add_argument('--ip', default='cnn_bram', help='HWH 中 AXI BRAM Controller 的实际名称')
    parser.add_argument('--image', required=True, help='裁好的单格图片，不是整张数独')
    parser.add_argument('--polarity', choices=['auto', 'dark_on_light', 'light_on_dark'], default='auto')
    parser.add_argument('--timeout', type=float, default=2.0)
    args = parser.parse_args()
    from pynq import Overlay

    overlay = Overlay(args.overlay, download=False)
    predictor = BramPredictor.from_overlay(overlay, args.ip, args.timeout)
    print(predictor.predict_image(args.image, args.polarity))


if __name__ == '__main__':
    main()
