# 电脑直接通过串口给 LCD 发一份 81 字节结果（不经过板子，用于单独测试 LCD）。
# 来源：dyx_lcd/send_board.ps1。唯一改动：默认结果文件改为 ..\hardware\lcd\actual_result.json
# 用法：.\send_board.ps1 -Port COM3
#       .\send_board.ps1 -Port COM3 -Result ..\results\live_xxxxxxxx\lcd_result.json
#       .\send_board.ps1 -Port COM3 -DryRun     # 只打印帧长度和校验和，不发送
param(
    [Parameter(Mandatory=$true)][string]$Port,
    [string]$Result = "$PSScriptRoot\..\hardware\lcd\actual_result.json",
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$displayResult = Get-Content -LiteralPath $Result -Raw | ConvertFrom-Json
[byte[]]$payload = $displayResult.display_bytes
if ($payload.Length -ne 81) { throw 'Expected 81 display bytes.' }
foreach ($value in $payload) {
    if (($value -band 0xE0) -ne 0 -or ($value -band 15) -gt 9) { throw 'Invalid cell byte.' }
}
$checksum = 0
foreach ($value in $payload) { $checksum = ($checksum + $value) -band 255 }
[byte[]]$packet = @(0x55,0xAA) + $payload + @([byte]$checksum)
if ($DryRun) { "Packet: $($packet.Length) bytes, checksum $checksum"; exit }
$serial = [System.IO.Ports.SerialPort]::new($Port,115200,[System.IO.Ports.Parity]::None,8,[System.IO.Ports.StopBits]::One)
$serial.ReadTimeout = 3000
$serial.WriteTimeout = 3000
try {
    $serial.Open()
    $serial.DiscardInBuffer()
    $serial.Write($packet,0,$packet.Length)
    $response = $serial.ReadByte()
    if ($response -eq 6) { 'ACK: LCD board updated.' }
    elseif ($response -eq 21) { throw 'NAK: LCD rejected the packet.' }
    else { throw "Unexpected response: $response" }
} finally {
    if ($serial.IsOpen) { $serial.Close() }
    $serial.Dispose()
}
