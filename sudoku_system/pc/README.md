# pc/ — 电脑端程序

这里的文件在**电脑**上运行，不需要拷到板子上（拷过去也没关系）。

| 文件 | 作用 |
|---|---|
| `lcd_bridge.py` | 一键运行时必须开着：板子经网线把 81 字节发来，它转成串口帧发给 LCD，再把 ACK 回给板子 |
| `send_board.ps1` | 不经过板子，电脑直接用串口给 LCD 发一份结果，用来单独测试 LCD |

> `lcd_bridge.py` 是按原接口重写的（原版不在 jupyter_notebooks 里）。如果你手上有原版，直接替换即可，板子端不用改。

## 第一次使用

1. 安装依赖：`pip install pyserial`
2. 电脑有线网卡设成固定 IP `192.168.2.1`，子网掩码 `255.255.255.0`；板子是 `192.168.2.99`
3. USB 线接板子的 PL UART，在设备管理器里看串口号（例如 `COM3`）
4. 放行防火墙（管理员 PowerShell，只放行板子这一个 IP）：

```powershell
New-NetFirewallRule -DisplayName "PYNQ LCD bridge" -Direction Inbound -Protocol TCP -LocalPort 8765 -RemoteAddress 192.168.2.99 -Action Allow
```

## 每次运行

```powershell
cd sudoku_system\pc
python lcd_bridge.py --port COM3
```

看到"LCD 桥已启动"后，再到板子上运行 `run.ipynb`。用完按 `Ctrl+C` 停止。

## 测试结束后

```powershell
Remove-NetFirewallRule -DisplayName "PYNQ LCD bridge"
```

## 常见问题

| 现象 | 原因 |
|---|---|
| 启动时报串口错误 | 串口号不对，或被串口助手等软件占用 |
| 启动时报地址错误 | 有线网卡 IP 不是 `192.168.2.1`；可以加 `--host` 指定 |
| 板子报 `PC LCD bridge is not ready` / 连接超时 | 桥没启动、网线没连、防火墙没放行 |
| 返回 504 / "LCD 没有回应" | 板子上还没加载 `lcd.bit`，或 USB 没接在 PL UART 上 |
| 返回 `ack: 21`（NAK） | 帧格式或校验错误，一般说明数据被改动过 |
