@echo off
REM Icarus Verilog 仿真（Windows）：双击或在 05_FPGA_Verilog 目录运行 scripts\run_sim.bat
cd /d "%~dp0.."
if not exist sim_out mkdir sim_out
cd sim_out
copy /Y ..\mem\*.mem . >nul
if exist vectors rmdir /S /Q vectors
xcopy /E /I /Y ..\tb\vectors vectors >nul
iverilog -g2005 -I ../rtl -o tb.vvp ../tb/tb_cnn_accel.v ../rtl/cnn_accel_top.v ../rtl/cnn_core.v ../rtl/cnn_axil_regs.v || goto :eof
vvp -n tb.vvp
pause
