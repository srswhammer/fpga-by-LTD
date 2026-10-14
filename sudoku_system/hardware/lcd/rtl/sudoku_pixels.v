module sudoku_pixels(input [10:0] x,input [10:0] y,input [647:0] board,output reg [23:0] color);
function [34:0] glyph;
 input [3:0] digit;
 begin
  case(digit)
   4'd0: glyph=35'b00000000000000000000000000000000000;
   4'd1: glyph=35'b00100011000010000100001000010001110;
   4'd2: glyph=35'b01110100010000100010001000100011111;
   4'd3: glyph=35'b11110000010000101110000010000111110;
   4'd4: glyph=35'b00010001100101010010111110001000010;
   4'd5: glyph=35'b11111100001000011110000010000111110;
   4'd6: glyph=35'b01110100001000011110100011000101110;
   4'd7: glyph=35'b11111000010001000100010000100001000;
   4'd8: glyph=35'b01110100011000101110100011000101110;
   4'd9: glyph=35'b01110100011000101111000010000101110;
   default: glyph=0;
  endcase
 end
endfunction
integer row,col,lx,ly,gx,gy,i;
reg [7:0] value;
reg [34:0] bits;
always @* begin
 color=24'hffffff;row=0;col=0;lx=0;ly=0;gx=0;gy=0;value=0;bits=0;
 if(x>=175 && x<=625 && y>=15 && y<=465) begin
  for(i=0;i<9;i=i+1) begin
   if(x>=175+i*50) col=i;
   if(y>=15+i*50) row=i;
  end
  lx=x-175-col*50;ly=y-15-row*50;
  if(x>=623 || y>=463 || lx==0 || ly==0 || (col%3==0 && lx<3) || (row%3==0 && ly<3))
   color=24'h000000;
  else begin
   value=board[(row*9+col)*8+:8];bits=glyph(value[3:0]);
   if(lx>=15 && lx<35 && ly>=11 && ly<39) begin
    gx=(lx-15)/4;gy=(ly-11)/4;
    if(bits[34-gy*5-gx]) color=value[4] ? 24'hff0000 : 24'h000000;
   end
  end
 end
end
endmodule
