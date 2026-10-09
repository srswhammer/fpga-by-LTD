# 板上 Python → BRAM → v2.4 CNN

当前交付的是板端 Python 接口代码和待硬件实现的协议，不是已上板运行的完整加速器。
仓库 `z15_camera_overlay` 使用 PYNQ，示例内核版本为 Python 3.8.2；因此本入口采用 PYNQ MMIO，不要求 PyTorch。
现有 `camera_capture.bit/.hwh` 只有相机通路，不能直接运行本入口。仓库 `module/rtl/cnn_top.v` 仍是旧版单卷积模型，也不能接上后当作 v2.4 使用。

## 数据怎么走

```text
相机 → 原有 VDMA → DDR 中的原始画面
                         ↓ 板上 Python：棋盘定位、校正、切成81格（外部程序负责）
                     单格图片
                         ↓ preprocess.py：灰度、裁剪、缩放、居中
                     uint8[28,28]
                         ↓ board_bram.py：196次32位MMIO写入
PS M_AXI_GP0 → AXI互连 → AXI BRAM Controller → 双口BRAM端口A
                                              端口B ↔ 控制状态机 + v2.4 CNN
                         ↑ Python轮询完成序号，读取9位识别结果
```

相机和显示屏仍可使用 DDR；这里只让预处理后的 CNN 输入走 BRAM，不改相机原有 VDMA。
Python 数组本身仍在 ARM 的内存中。MMIO 把其中784个像素复制到 PL BRAM；不是把PNG文件存入BRAM，也不是用Python变量名让PL访问DDR。
每次处理一格，81格顺序复用同一块BRAM；不必把81格全部塞进BRAM。先用单缓冲，尚不实现双缓冲并行。

## Vivado 连接约定（需要硬件端落实）

1. 在包含相机及后续显示通路的同一个完整工程中增加 AXI BRAM Controller，建议命名 `cnn_bram`。接 PS 的 GP 主接口；通过互连与现有 VDMA 控制接口共享。
2. 控制器使用32位数据宽度、单BRAM端口模式，不启用ECC；只占双口BRAM的端口A。BRAM采用真双口、每端口32位、深度1024字，共4096字节。端口B留给CNN接口状态机。
3. 本协议首版要求两端使用同一AXI时钟，复位同步释放。端口B是同步读存储器，RTL需按实际配置处理读延迟，不能照搬旧模块的异步数组读取。
4. 状态机经端口B读取输入、启动正确的v2.4 CNN，再把结果写回BRAM；状态机和CNN如共用端口B，必须有明确仲裁。不要让AXI控制器同时占用A/B两个端口。
5. BRAM基地址由Vivado Address Editor分配，通过配套HWH交给Python；代码没有硬编码物理地址。表中的偏移是这次新定义的协议，不是已有硬件寄存器。

4096字节是接口缓冲区的逻辑容量，不是整个CNN的BRAM占用。权重、中间特征和相机/显示缓存另计；综合资源和实际速度尚未验证。

## BRAM内容与所有权

所有字段为32位小端字；下表偏移单位是**字节**。PL端若连接使用字地址的RAM端口，地址要除以4，不能直接把字节偏移当字地址。

| 偏移 | 字段 | 运行时写入者 | 内容 |
|---|---|---|---|
| 0x00 | MAGIC | PL | 固定0x434E4E32，复位初始化最后写入 |
| 0x04 | VERSION | PL | 协议版本1 |
| 0x08 | MODEL | PL | v2.4当前权重标签0xEE067A81 |
| 0x0C | STATUS | PL | READY=1；BUSY=2；FAULT=4，三者互斥 |
| 0x10 | REQUEST | PS | 本次请求序号，每次在ACK基础上加1，32位回绕 |
| 0x14 | ACK | PL | 已完成序号；必须在结果、错误码和状态写好后最后更新 |
| 0x18 | RESULT | PL | 低9位为独热输出，高23位为0；空白=0 |
| 0x1C | ERROR | PL | 成功为0，失败为硬件端定义的非零错误码 |
| 0x20～0xFF | 保留 | — | 本版不用，不存权重 |
| 0x100～0x40F | PIXELS | PS | 784字节，28×28行主序UINT8像素 |
| 0x410～0xFFF | 保留 | — | 本版不用 |

像素 `p[4*i]` 放在第i字的bit[7:0]，后续三个像素依次放bit[15:8]、[23:16]、[31:24]。
例如像素01、02、03、04写成整数0x04030201。第r行c列像素的字节偏移是 `0x100+r*28+c`，不转置，不除255/256，不改成有符号INT8。

模型标签取当前 `weights/int8.npz` SHA256的前8位，只用于区分构建版本，不是自动校验RTL计算正确性的机制。完整SHA256：
`ee067a815732ea1618fb0f12b5220837e58a7386d5505946b2ad8179c603cee2`。
硬件仍需实现两层卷积1→4→8、两次池化、Flatten200和FC10；卷积1/2右移分别为9/7，四舍五入偏置及饱和规则以 `predict.py` 和 `weights/int8.json` 为准。不能只把旧硬件标识改成新值。
权重由硬件工程按现有参数构建/初始化；本脚本只传像素，不向BRAM上传模型权重。

## 状态机握手顺序

1. **复位初始化**：PL先使MAGIC无效，初始化REQUEST=ACK=0、RESULT=ERROR=0、VERSION/MODEL和STATUS=READY，最后写MAGIC。仅此初始化阶段PL可写REQUEST；运行中它只读REQUEST。PS必须等初始化结束后创建实例。
2. **写入输入**：PS检查标识、STATUS=READY且REQUEST=ACK，写入196个像素字，回读最后一个字后才提交新的REQUEST。PL空闲时只轮询REQUEST，不读取正在写入的像素。
3. **启动计算**：PL发现REQUEST≠ACK，锁存请求序号，置BUSY，读取784字节并运行CNN。PS不得在此期间写下一格。不能只等待单周期start或done让Python采样。
4. **发布结果**：PL写RESULT、ERROR=0、STATUS=READY，最后写ACK为锁存的请求序号。写READY到写ACK之间，PL仍处于完成发布状态，不得把同一请求重复启动；ACK发布后才返回轮询状态。
5. **读取结果**：PS看到ACK等于本次序号，再读取状态、错误码、结果。由于ACK持续保存，即使Python被Linux调度延迟也不会漏掉完成通知。
6. **失败处理**：PL若检测到计算错误，写非零ERROR、STATUS=FAULT，再发布ACK；Python抛出错误。软件等待超过默认2秒也报错，不自动重发、清零或继续覆盖BRAM。检查并复位硬件后重建实例；2秒是保护超时，不是性能指标。

复位或重新加载Overlay期间禁止调用此接口。同一硬件仅由一个进程、一个实例控制；类内锁仅防止同一实例的多线程同时覆盖输入，不提供跨进程仲裁。BRAM内容掉电丢失，不是永久保存图片的介质。

## 板上如何调用

板上实时路径只用 `board_bram.py` 和 `preprocess.py`，依赖现有PYNQ、NumPy、Pillow。不要在板上直接安装面向Windows软件参考的 `requirements.txt`，也不需要安装Torch。已做Python3.8兼容调整，但本次未在真实板上执行；先检查板上能导入这三个库。

在Jupyter中复用已经加载的完整Overlay对象：

```python
from board_bram import BramPredictor

# overlay 是外部程序已加载的“相机 + CNN”完整Overlay，不是旧camera_capture.bit。
cnn = BramPredictor.from_overlay(overlay, ip_name='cnn_bram')

# cell_gray 是外部定位/校正/分格程序给出的一个格子，黑字浅底。
code = cnn.predict_image(cell_gray, polarity='dark_on_light')
print(code)  # 例如数字3为000000100；这里只是格式举例，不是测试输出

# 已经预处理为白字黑底28×28 UINT8时，直接走这一入口，不再预处理。
code = cnn.predict_uint8(normalized_uint8)

# cells 为按棋盘行主序排列的81个单格；逐格计算，不重新加载模型或Overlay。
codes = [cnn.predict_image(cell, polarity='dark_on_light') for cell in cells]
digits = [int(code, 2).bit_length() for code in codes]  # 空白0，数字1～9，交给求解器
```

浅底黑字照片显式指定dark_on_light，避免曝光较暗时auto误判极性；已经标准化的白字黑底输入应走predict_uint8，或predict_image配light_on_dark。预处理不负责定位整张棋盘或去除粗网格线。

命令行（相应完整Overlay必须已加载，以下bit文件名只是示例）：

```bash
python3 -B board_bram.py --overlay /home/xilinx/jupyter_notebooks/sudoku_system.bit --ip cnn_bram --image cell.png --polarity dark_on_light
```

命令只读取匹配HWH并绑定已加载Overlay，不自动重新烧写PL，不中断摄像头。实际IP名若不同，通过`--ip`传入；没有该IP或协议标识不对就停止，不退回CPU预测。不要把未知地址填入MMIO试写。

实时运行无须写PNG/HEX或再读文件。`预处理图像/`保留作离线仿真及上板对照。先送一张，再检查该目录270张输入与`软件参考结果.txt`输出逐个一致；其中那张4的软件本来就判9，硬件应同样输出9。还应在RTL仿真对比十类整数分数，检查计算一致性，而不只看最终类别。

## 当前验证边界及依据

本次可在电脑验证预处理字节不变、32位打包顺序、模拟MMIO握手、超时和独热格式；不能据此认定AXI时序、BRAM读延迟、板上Python环境、CNN算术或整机速度已通过。
2026-10-09电脑端实测：9项回归测试全部通过；270张原始照片在auto和dark_on_light两种模式下，修改前后预处理像素逐字节一致；270张标准输入的模拟MMIO传输字节一致，软件INT8仍为269/270。模型权重、TXT参数、predict.py的SHA256均未改变；板端模块导入不加载Torch。仅做了Python3.8语法检查，未在Python3.8真实运行时执行。
测试脚本和结果保存在原模型的 `evaluation/板端BRAM验证_20261009/`，没有把测试代码塞入交接包。
下一步硬件端需实现上述协议状态机、正确的v2.4 CNN及完整Overlay，生成匹配bit/hwh后才能做真实上板验收。

- [PYNQ MMIO](https://pynq.readthedocs.io/en/v2.7.0/pynq_package/pynq.mmio.html)：通过内存映射读写PL地址空间。
- [PYNQ Overlay](https://pynq.readthedocs.io/en/v2.7.0/pynq_package/pynq.overlay.html)：IP字典提供phys_addr/addr_range，download=False不下载bitstream。
- [AMD AXI BRAM Controller PG078](https://docs.amd.com/v/u/en-US/pg078-axi-bram-ctrl)：控制器可配置为仅使用一个BRAM端口。
