"""PYNQ 2.7 端 DigitCNN FPGA 驱动（Zynq-7015）。

用法（板上 Jupyter）：
    from cnn_fpga_driver import CnnFpga
    acc = CnnFpga("cnn_digit.bit")
    pred, scores = acc.infer(img_u8_28x28)       # img 来自 preprocess_digit_image()
    acc.self_test("hardware_reference.npz")      # 固定十张与 CPU 标准答案逐位比对

★ 下面“待确认”常量必须与 Vivado Block Design 中的实例名 / cnn_defines.vh 保持一致。
"""
import time
import numpy as np
from pynq import Overlay, allocate

# ---------------------------------------------------------------------------
# [待确认] Block Design 实例名（Overlay 中 ip_dict 的键名）
# ---------------------------------------------------------------------------
BITSTREAM     = "cnn_digit.bit"      # 同目录需要同名 .hwh
IP_CNN        = "cnn_accel_0"        # cnn_accel_top 的实例名
IP_DMA        = "axi_dma_0"          # AXI DMA 实例名

# ---------------------------------------------------------------------------
# [待确认] 与 rtl/cnn_defines.vh 同步
# ---------------------------------------------------------------------------
USE_AXI_LITE    = True               # CNN_USE_AXI_LITE
USE_M_AXIS      = True               # CNN_USE_M_AXIS
OUT_ARGMAX_WORD = True               # CNN_OUT_ARGMAX_WORD
OUT_WORDS       = 11 if OUT_ARGMAX_WORD else 10

REG_CTRL   = 0x00
REG_STATUS = 0x04
REG_RESULT = 0x08
REG_CYCLES = 0x0C
REG_FRAMES = 0x10
REG_ID     = 0x14
REG_SCORE0 = 0x40
IP_ID      = 0x434E4E01

ST_BUSY, ST_DONE, ST_ERR, ST_IDLE = 0x1, 0x2, 0x4, 0x8


class CnnFpga:
    def __init__(self, bitstream=BITSTREAM, fclk_mhz=None):
        self.ol = Overlay(bitstream)
        self.dma = getattr(self.ol, IP_DMA)
        self.ip = getattr(self.ol, IP_CNN) if USE_AXI_LITE else None
        self.fclk_mhz = fclk_mhz or self._read_fclk()
        # DMA 缓冲区（连续物理内存）：输入 784 字节，输出 OUT_WORDS 个 int32
        self.in_buf = allocate(shape=(784,), dtype=np.uint8)
        self.out_buf = allocate(shape=(OUT_WORDS,), dtype=np.int32)
        if self.ip is not None:
            ident = self.ip.read(REG_ID)
            if ident != IP_ID:
                raise RuntimeError(f"IP ID 不匹配：读到 0x{ident:08x}，期望 0x{IP_ID:08x}（bit 版本或地址错误）")
            self.soft_reset()

    @staticmethod
    def _read_fclk():
        try:
            from pynq.ps import Clocks
            return Clocks.fclk0_mhz
        except Exception:
            return 100.0

    # ------------------------------------------------------------------ 控制
    def soft_reset(self):
        self.ip.write(REG_CTRL, 0x1)
        self.ip.write(REG_STATUS, ST_DONE | ST_ERR)   # W1C 清粘滞位

    def status(self):
        s = self.ip.read(REG_STATUS)
        return {"busy": bool(s & ST_BUSY), "done": bool(s & ST_DONE),
                "tlast_err": bool(s & ST_ERR), "idle": bool(s & ST_IDLE)}

    # ------------------------------------------------------------------ 推理
    def infer(self, img_u8):
        """img_u8: uint8[28,28]（白字黑底，preprocess 的 normalized_uint8）→ (pred, scores[int32×10])"""
        img = np.ascontiguousarray(img_u8, dtype=np.uint8).reshape(784)
        self.in_buf[:] = img
        # PYNQ 2.7 的 allocate 缓冲区默认 cacheable，transfer 内部会 flush/invalidate
        if USE_M_AXIS:
            self.dma.recvchannel.transfer(self.out_buf)   # 先挂接收，再发送
        self.dma.sendchannel.transfer(self.in_buf)
        self.dma.sendchannel.wait()
        if USE_M_AXIS:
            self.dma.recvchannel.wait()
            scores = np.array(self.out_buf[:10], dtype=np.int32)
            pred = int(self.out_buf[10]) if OUT_ARGMAX_WORD else int(scores.argmax())
        else:
            self._wait_done()
            scores = np.array([self._s32(self.ip.read(REG_SCORE0 + 4 * i)) for i in range(10)], dtype=np.int32)
            pred = self.ip.read(REG_RESULT) & 0xF
        if self.ip is not None:
            self.ip.write(REG_STATUS, ST_DONE)
        return pred, scores

    def _wait_done(self, timeout_s=0.1):
        t0 = time.time()
        while not (self.ip.read(REG_STATUS) & ST_DONE):
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"CNN 未完成：{self.status()}")

    @staticmethod
    def _s32(v):
        return v - (1 << 32) if v & 0x80000000 else v

    def last_cycles(self):
        return self.ip.read(REG_CYCLES) if self.ip is not None else None

    # ------------------------------------------------------------------ 自测
    def self_test(self, ref_path="hardware_reference.npz"):
        ref = np.load(ref_path, allow_pickle=False)
        ok = 0
        for n, img in enumerate(ref["fixed_inputs_uint8"]):
            t0 = time.perf_counter()
            pred, scores = self.infer(img)
            dt = (time.perf_counter() - t0) * 1e6
            same = np.array_equal(scores, ref["scores_int32"][n]) and pred == ref["predictions_int64"][n]
            ok += same
            cyc = self.last_cycles()
            hw_us = f"{cyc / self.fclk_mhz:.1f} µs" if cyc else "-"
            print(f"图{n}: 预测 {pred} 期望 {ref['predictions_int64'][n]}  分数一致={same}  "
                  f"PL 计算 {cyc} 拍 ≈ {hw_us}  端到端 {dt:.0f} µs")
        print(f"==== {ok}/10 与 CPU 整数标准答案逐位一致 ====")
        return ok == 10


if __name__ == "__main__":
    acc = CnnFpga()
    acc.self_test()
