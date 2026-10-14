set here [file dirname [file normalize [info script]]]
cd $here
file mkdir build
read_verilog [glob rtl/*.v]
set_property include_dirs [list $here/rtl] [current_fileset]
read_xdc lcd.xdc
synth_design -top sudoku_uart_top -part xc7z015clg485-2
opt_design
write_checkpoint -force build/synth.dcp
place_design
phys_opt_design
route_design
report_timing_summary -file build/timing.txt
report_timing -delay_type min -max_paths 5 -file build/hold_paths.txt
report_utilization -file build/utilization.txt
write_checkpoint -force build/lcd_display.dcp
set setup_slack [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold_slack [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
if {$setup_slack < 0 || $hold_slack < 0} {error "Timing failed: setup $setup_slack hold $hold_slack"}
write_bitstream -force build/lcd_display.bit
