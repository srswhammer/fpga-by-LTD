"""电脑端 LCD 串口桥：板子通过网线发 HTTP 请求，本程序把 81 字节打成串口帧发给 LCD 的 PL UART，
再把 LCD 回的 ACK/NAK 返回给板子。

说明：原来的 lcd_bridge.py 不在 jupyter_notebooks 里。本文件是按原 README 和板子端代码
（sudoku/lcd.py）里用到的接口重写的；如果你手上有原版，直接用原版替换也可以。

接口（与板子端 sudoku/lcd.py 对应）：
    GET  /health                           → {"ready": true}
    POST /display {"display_bytes": [81 个整数]} → {"ack": 6}   6=ACK，21=NAK
串口帧（与 send_board.ps1、LCD 里的 board_bridge.v 相同）：
    55 AA + 81 字节 + 81 字节求和的低 8 位；115200、8N1
    每字节：bit7～5 = 0，bit4 = 0 原题 / 1 补入，bit3～0 = 数字 0～9
安全：只监听电脑有线地址（默认 192.168.2.1），只接受板子 IP（默认 192.168.2.99）的请求。

依赖：pip install pyserial
用法：python lcd_bridge.py --port COM3          （串口号以设备管理器里看到的为准）
停止：Ctrl+C
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ACK, NAK = 0x06, 0x15


def build_packet(payload):
    """检查 81 个显示字节并加上帧头和校验和。"""
    if not isinstance(payload, list) or len(payload) != 81:
        raise ValueError('display_bytes 必须是 81 个整数')
    for value in payload:
        if not isinstance(value, int) or value < 0 or value > 255:
            raise ValueError('display_bytes 里有不是 0～255 的值：%r' % (value,))
        if value & 0xE0 or (value & 0x0F) > 9:
            raise ValueError('无效的格子字节：0x%02X' % value)
    return bytes([0x55, 0xAA] + payload + [sum(payload) & 0xFF])


class SerialLink:
    """每次发送时打开串口、发完关闭（和 send_board.ps1 的做法一样），多个请求排队发。"""

    def __init__(self, port, baud=115200, timeout=3.0, serial_factory=None):
        self.port, self.baud, self.timeout = port, baud, timeout
        self.lock = threading.Lock()
        if serial_factory is None:
            import serial                        # pyserial
            serial_factory = serial.Serial
        self.serial_factory = serial_factory

    def check(self):
        """启动时试开一次串口，串口号写错能马上发现。"""
        with self.lock:
            self.serial_factory(self.port, self.baud, timeout=self.timeout).close()

    def send(self, packet):
        """发送一帧，返回 LCD 回的那个字节；超时返回 None。"""
        with self.lock:
            link = self.serial_factory(self.port, self.baud, timeout=self.timeout,
                                       write_timeout=self.timeout)
            try:
                link.reset_input_buffer()
                link.write(packet)
                link.flush()
                reply = link.read(1)
            finally:
                link.close()
        return reply[0] if reply else None


def make_handler(link, allowed_clients):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _allowed(self):
            if allowed_clients and self.client_address[0] not in allowed_clients:
                self._reply(403, {'error': 'client not allowed'})
                return False
            return True

        def do_GET(self):
            if not self._allowed():
                return
            if self.path == '/health':
                self._reply(200, {'ready': True, 'port': link.port})
            else:
                self._reply(404, {'error': 'not found'})

        def do_POST(self):
            if not self._allowed():
                return
            if self.path != '/display':
                self._reply(404, {'error': 'not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if length <= 0 or length > 4096:
                    raise ValueError('请求体长度不对')
                body = json.loads(self.rfile.read(length).decode('utf-8'))
                packet = build_packet(body.get('display_bytes'))
            except (ValueError, AttributeError) as error:
                self._reply(400, {'error': str(error)})
                return
            try:
                ack = link.send(packet)
            except Exception as error:                 # 串口被占用、拔掉等
                self._reply(502, {'error': 'serial: %s' % error})
                return
            if ack is None:
                self._reply(504, {'error': 'LCD 没有回应（串口超时）'})
                return
            print('已发送 81 字节，LCD 回复 0x%02X（%s）' % (ack, 'ACK' if ack == ACK else
                                                         'NAK' if ack == NAK else '未知'))
            self._reply(200, {'ack': ack})

        def log_message(self, fmt, *args):            # 只打印简短日志
            print('%s  %s' % (self.client_address[0], fmt % args))

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description='PYNQ 数独 LCD 串口桥')
    parser.add_argument('--port', required=True, help='LCD PL UART 对应的串口，例如 COM3')
    parser.add_argument('--host', default='192.168.2.1', help='只监听这个地址（电脑有线网卡 IP）')
    parser.add_argument('--http-port', type=int, default=8765)
    parser.add_argument('--allow', default='192.168.2.99',
                        help='只接受这些 IP 的请求，逗号分隔；写 any 表示不限制')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--timeout', type=float, default=3.0, help='等 LCD 回 ACK 的秒数')
    args = parser.parse_args(argv)

    link = SerialLink(args.port, args.baud, args.timeout)
    link.check()
    allowed = None if args.allow == 'any' else {ip.strip() for ip in args.allow.split(',')}
    server = HTTPServer((args.host, args.http_port), make_handler(link, allowed))
    print('LCD 桥已启动：http://%s:%d  串口 %s  允许 %s' % (
        args.host, args.http_port, args.port, args.allow))
    print('按 Ctrl+C 停止')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print('已停止')


if __name__ == '__main__':
    main()
