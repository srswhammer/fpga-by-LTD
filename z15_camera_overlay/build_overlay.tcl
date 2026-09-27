set o [file normalize [file dirname [info script]]]
create_project camera_capture "$o/vivado" -part xc7z015clg485-2 -force
add_files [list "$o/camera_frontend.v" "$o/i2c_dri.v" "$o/i2c_ov5640_rgb565_cfg.v"]
add_files -fileset constrs_1 "$o/camera_overlay.xdc"
create_bd_design camera_capture
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable"} [get_bd_cells ps7]
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_EN_CLK1_PORT {1} CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ {50}] [get_bd_cells ps7]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_vdma:* vdma
set_property -dict [list CONFIG.c_include_mm2s {0} CONFIG.c_include_s2mm {1} CONFIG.c_include_sg {0} CONFIG.c_num_fstores {1} CONFIG.c_m_axi_s2mm_data_width {64} CONFIG.c_s2mm_linebuffer_depth {4096}] [get_bd_cells vdma]
create_bd_cell -type module -reference camera_frontend frontend
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/ps7/M_AXI_GP0" Clk "Auto"} [get_bd_intf_pins vdma/S_AXI_LITE]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* hp_sc
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells hp_sc]
connect_bd_intf_net [get_bd_intf_pins vdma/M_AXI_S2MM] [get_bd_intf_pins hp_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins hp_sc/M00_AXI] [get_bd_intf_pins ps7/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins frontend/m_axis] [get_bd_intf_pins vdma/S_AXIS_S2MM]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins vdma/m_axi_s2mm_aclk]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins hp_sc/aclk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins hp_sc/aresetn]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ps7/S_AXI_HP0_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK1] [get_bd_pins frontend/clk_50m]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins frontend/resetn]
foreach n {cam_pclk cam_vsync cam_href cam_data cam_rst_n cam_pwdn cam_scl cam_sda led_init led_pclk} {
 make_bd_pins_external [get_bd_pins frontend/$n]
 set_property name $n [get_bd_ports ${n}_0]
}
connect_bd_net [get_bd_ports cam_pclk] [get_bd_pins vdma/s_axis_s2mm_aclk]
assign_bd_address -target_address_space [get_bd_addr_spaces vdma/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_HP0/HP0_DDR_LOWOCM]
validate_bd_design
save_bd_design
generate_target all [get_files "$o/vivado/camera_capture.srcs/sources_1/bd/camera_capture/camera_capture.bd"]
make_wrapper -files [get_files "$o/vivado/camera_capture.srcs/sources_1/bd/camera_capture/camera_capture.bd"] -top
add_files -norecurse "$o/vivado/camera_capture.gen/sources_1/bd/camera_capture/hdl/camera_capture_wrapper.v"
set_property top camera_capture_wrapper [current_fileset]
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "Implementation failed" }
file copy -force "$o/vivado/camera_capture.runs/impl_1/camera_capture_wrapper.bit" "$o/camera_capture.bit"
file copy -force "$o/vivado/camera_capture.gen/sources_1/bd/camera_capture/hw_handoff/camera_capture.hwh" "$o/camera_capture.hwh"
