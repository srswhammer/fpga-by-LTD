module sudoku_uart_top(
 input sys_clk,input sys_rst_n,input uart_rxd,output uart_txd,
 output lcd_de,output lcd_hs,output lcd_vs,output lcd_bl,
 output lcd_clk,output lcd_rst,inout [23:0] lcd_rgb,output ms7200_rst_n);
wire [15:0] lcd_id=16'h4384; // Verified ATK 4.3-inch 800x480 target.
wire lcd_pclk;
wire [10:0] x,y,hdisp,vdisp;
wire [23:0] color,rgb_out;
wire rx_done,tx_en,tx_busy;
wire [7:0] rx_data,tx_data;
wire [647:0] board;
assign lcd_rgb=lcd_de ? rgb_out : 24'bz;
assign ms7200_rst_n=0;
clk_div u_clk_div(.clk(sys_clk),.rst_n(sys_rst_n),.lcd_id(lcd_id),.lcd_pclk(lcd_pclk));
uart_rx u_rx(.clk(sys_clk),.rst_n(sys_rst_n),.uart_rxd(uart_rxd),.uart_rx_done(rx_done),.uart_rx_data(rx_data));
uart_tx u_tx(.clk(sys_clk),.rst_n(sys_rst_n),.uart_tx_en(tx_en),.uart_tx_data(tx_data),.uart_txd(uart_txd),.uart_tx_busy(tx_busy));
board_bridge u_bridge(.clk(sys_clk),.pclk(lcd_pclk),.rst_n(sys_rst_n),
 .rx_done(rx_done),.rx_data(rx_data),.frame_blank(x==0 && y==0),
 .tx_en(tx_en),.tx_data(tx_data),.board(board));
sudoku_pixels u_pixels(.x(x),.y(y),.board(board),.color(color));
lcd_driver u_driver(.lcd_pclk(lcd_pclk),.rst_n(sys_rst_n),.lcd_id(lcd_id),
 .pixel_data(color),.pixel_xpos(x),.pixel_ypos(y),.h_disp(hdisp),.v_disp(vdisp),
 .data_req(),.lcd_de(lcd_de),.lcd_hs(lcd_hs),.lcd_vs(lcd_vs),.lcd_bl(lcd_bl),
 .lcd_clk(lcd_clk),.lcd_rst(lcd_rst),.lcd_rgb(rgb_out));
endmodule
