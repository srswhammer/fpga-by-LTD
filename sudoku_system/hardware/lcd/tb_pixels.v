`timescale 1ns/1ps
module tb_pixels;
`include "board_init.vh"
reg [10:0] x=0,y=0;
wire [23:0] color;
sudoku_pixels dut(x,y,INITIAL_BOARD,color);
initial begin
 #1;if(color!=24'hffffff) $fatal(1,"background");
 x=175;y=15;#1;if(color!=0) $fatal(1,"grid border");
 x=190;y=26;#1;if(color!=0) $fatal(1,"original 5 not black");
 x=302;y=26;#1;if(color!=24'hff0000) $fatal(1,"filled 4 not red");
 $display("PASS: white background, grid, original black, filled red");$finish;
end
endmodule
