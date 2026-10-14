`timescale 1ns/1ps
module tb_bridge;
reg clk=0,pclk=0,rst=0,rx_done=0,blank=0;
reg [7:0] data=0;
wire tx_en;wire [7:0] tx_data;wire [647:0] board;
reg [647:0] saved_board;
integer i,ack=0,nak=0;
always #10 clk=~clk;
always #20 pclk=~pclk;
board_bridge #(.TIMEOUT(200)) dut(clk,pclk,rst,rx_done,data,blank,tx_en,tx_data,board);
always @(posedge clk) if(tx_en) begin
 if(tx_data==6) ack=ack+1;
 if(tx_data==21) nak=nak+1;
end
task byte_in;
 input [7:0] b;
 begin @(negedge clk);rx_done=1;data=b;@(negedge clk);rx_done=0;end
endtask
initial begin
 #25;rst=1;#100;saved_board=board;
 byte_in(8'h55);byte_in(8'haa);
 for(i=0;i<81;i=i+1) byte_in(8'h11);
 byte_in(8'h61); // 81*0x11 modulo 256
 #200;
 if(board!==saved_board || ack!=0) $fatal(1,"updated saved_board blank");
 blank=1;#240;blank=0;
 for(i=0;i<81;i=i+1) if(board[i*8+:8]!=8'h11) $fatal(1,"board mismatch");
 if(ack!=1) $fatal(1,"no ACK");
 byte_in(8'h55);byte_in(8'haa);
 for(i=0;i<81;i=i+1) byte_in(8'h01);
 byte_in(8'h00);#100;
 if(nak!=1 || board[7:0]!=8'h11) $fatal(1,"bad checksum accepted");
 byte_in(8'h55);byte_in(8'haa);
 for(i=0;i<81;i=i+1) byte_in(8'h20);
 byte_in(8'h20);#100;
 if(nak!=2 || board[7:0]!=8'h11) $fatal(1,"reserved bits accepted");
 byte_in(8'h55);byte_in(8'haa);#4500;
 if(nak!=3) $fatal(1,"timeout missing");
 $display("PASS: atomic update, ACK, checksum/reserved-bit rejection, timeout");$finish;
end
initial begin #100000;$fatal(1,"test timeout");end
endmodule

