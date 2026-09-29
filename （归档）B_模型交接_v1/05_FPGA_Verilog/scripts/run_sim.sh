#!/bin/sh
# Icarus Verilog 仿真：在 05_FPGA_Verilog 目录下运行  sh scripts/run_sim.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p sim_out && cd sim_out
cp ../mem/*.mem .
rm -rf vectors && cp -r ../tb/vectors .
iverilog -g2005 -I ../rtl -o tb.vvp ../tb/tb_cnn_accel.v ../rtl/cnn_accel_top.v ../rtl/cnn_core.v ../rtl/cnn_axil_regs.v
vvp -n tb.vvp
