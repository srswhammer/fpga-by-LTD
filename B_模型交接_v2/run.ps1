# 优先使用包内环境；本机可以复用已经配置好的 DigitCNN 环境。
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$localPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$sharedPython = Join-Path (Split-Path $PSScriptRoot -Parent) 'DigitCNN/01_环境与图片处理/.venv/Scripts/python.exe'
if (Test-Path -LiteralPath $localPython) {
    $pythonExecutable = $localPython
} elseif (Test-Path -LiteralPath $sharedPython) {
    $pythonExecutable = $sharedPython
} else {
    $pythonExecutable = (Get-Command python -ErrorAction Stop).Source
}
& $pythonExecutable -B (Join-Path $PSScriptRoot 'predict.py') @args
exit $LASTEXITCODE
