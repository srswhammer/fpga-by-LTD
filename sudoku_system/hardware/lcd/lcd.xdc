#时序约束
create_clock -period 20.000 -name sys_clk [get_ports sys_clk]

#IO管脚约束
set_property -dict {PACKAGE_PIN Y14 IOSTANDARD LVCMOS33} [get_ports sys_clk]
set_property -dict {PACKAGE_PIN G2 IOSTANDARD LVCMOS18} [get_ports sys_rst_n]

#--------------------------------RGB LCD------------------------------------------
set_property -dict {PACKAGE_PIN W17 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[0]}]
set_property -dict {PACKAGE_PIN Y17 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[1]}]
set_property -dict {PACKAGE_PIN U16 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[2]}]
set_property -dict {PACKAGE_PIN T16 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[3]}]
set_property -dict {PACKAGE_PIN Y13 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[4]}]
set_property -dict {PACKAGE_PIN Y12 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[5]}]
set_property -dict {PACKAGE_PIN AB14 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[6]}]
set_property -dict {PACKAGE_PIN AB13 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[7]}]
set_property -dict {PACKAGE_PIN AA20 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[8]}]
set_property -dict {PACKAGE_PIN AA19 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[9]}]
set_property -dict {PACKAGE_PIN W16 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[10]}]
set_property -dict {PACKAGE_PIN V16 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[11]}]
set_property -dict {PACKAGE_PIN AB19 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[12]}]
set_property -dict {PACKAGE_PIN AB18 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[13]}]
set_property -dict {PACKAGE_PIN V18 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[14]}]
set_property -dict {PACKAGE_PIN W18 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[15]}]
set_property -dict {PACKAGE_PIN W12 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[16]}]
set_property -dict {PACKAGE_PIN W13 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[17]}]
set_property -dict {PACKAGE_PIN V15 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[18]}]
set_property -dict {PACKAGE_PIN W15 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[19]}]
set_property -dict {PACKAGE_PIN V13 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[20]}]
set_property -dict {PACKAGE_PIN V14 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[21]}]
set_property -dict {PACKAGE_PIN AA14 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[22]}]
set_property -dict {PACKAGE_PIN AA15 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[23]}]

set_property -dict {PACKAGE_PIN A5 IOSTANDARD LVCMOS18} [get_ports lcd_hs]
set_property -dict {PACKAGE_PIN A4 IOSTANDARD LVCMOS18} [get_ports lcd_vs]
set_property -dict {PACKAGE_PIN U18 IOSTANDARD LVCMOS33} [get_ports lcd_de]
set_property -dict {PACKAGE_PIN A6 IOSTANDARD LVCMOS18} [get_ports lcd_bl]
set_property -dict {PACKAGE_PIN U17 IOSTANDARD LVCMOS33} [get_ports lcd_clk]
set_property -dict {PACKAGE_PIN A7 IOSTANDARD LVCMOS18} [get_ports lcd_rst]
set_property -dict {PACKAGE_PIN G3 IOSTANDARD LVCMOS18} [get_ports ms7200_rst_n]

set_property -dict {PACKAGE_PIN AA11 IOSTANDARD LVCMOS33} [get_ports uart_rxd]
set_property -dict {PACKAGE_PIN AB11 IOSTANDARD LVCMOS33} [get_ports uart_txd]
create_generated_clock -name lcd25 -source [get_ports sys_clk] -divide_by 2 [get_pins u_clk_div/clk_25m_reg/Q]
set_false_path -to [get_pins u_rx/uart_rxd_d0_reg/D]
set_false_path -to [get_pins u_bridge/req_sync1_reg/D]
set_false_path -to [get_pins u_bridge/ack_sync1_reg/D]
# Staging is held unchanged until the synchronized LCD apply acknowledgement.
# Capture follows two request synchronizers; allow at most one pixel period
# of bus delay, without imposing same-edge hold on this bundled CDC transfer.
set_max_delay -datapath_only 40 -from [get_cells {u_bridge/staging_reg[*]}] -to [get_cells {u_bridge/board_reg[*]}]
set_false_path -hold -from [get_cells {u_bridge/staging_reg[*]}] -to [get_cells {u_bridge/board_reg[*]}]
