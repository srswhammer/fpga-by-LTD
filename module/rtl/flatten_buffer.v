`timescale 1ns/1ps
//
// flatten_buffer: 676 字节（4x13x13 uint8）池化后展平结果缓存。
// 地址 = channel*169 + row*13 + column，与 integer_inference.py 的
// flatten 规则完全一致；卷积/池化单元写入，全连接单元读出。
//
module flatten_buffer (
    input  wire       clk,

    input  wire        we,
    input  wire [9:0]  waddr,   // 0..675
    input  wire [7:0]  wdata,

    input  wire [9:0]  raddr,   // 0..675
    output wire [7:0]  rdata
);

    reg [7:0] mem [0:675];

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
    end

    assign rdata = mem[raddr];

endmodule
