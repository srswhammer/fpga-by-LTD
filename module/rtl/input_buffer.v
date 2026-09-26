`timescale 1ns/1ps
//
// input_buffer: 784 字节（28x28 uint8）输入图片缓存。
// 行主序存储，列变化最快，与 integer_inference.py 的输入布局一致：
//   字节地址 = row*28 + col
//
module input_buffer (
    input  wire       clk,

    // 写端口：装载图片，一次一个字节
    input  wire        we,
    input  wire [9:0]  waddr,   // 0..783
    input  wire [7:0]  wdata,

    // 读端口：按 (行,列) 组合寻址，异步读（LUTRAM 风格，容量小无需 BRAM）
    input  wire [4:0]  rrow,    // 0..27
    input  wire [4:0]  rcol,    // 0..27
    output wire [7:0]  rdata
);

    reg [7:0] mem [0:783];

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
    end

    wire [9:0] raddr = rrow * 7'd28 + rcol;
    assign rdata = mem[raddr];

endmodule
