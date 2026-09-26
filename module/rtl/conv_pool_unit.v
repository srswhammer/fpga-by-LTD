`timescale 1ns/1ps
//
// conv_pool_unit：卷积 + ReLU/右移量化 + 2x2 最大池化，逐层对齐
// B_模型交接_v1/02_模型核心/integer_inference.py 与
// B_模型交接_v1/03_FPGA交接/接口说明.md 的数值合同：
//
//   conv_acc  = bias_int32 + sum(input_uint8 * weight_int8)   (互相关，stride=1, padding=0)
//   relu_u8   = sat_u8( (max(conv_acc,0) + 128) >> 8 )        (右移位数 = 14-6 = 8)
//   pool_u8   = 每通道 2x2、stride=2 的无符号最大值
//   展平地址  = channel*169 + row*13 + column
//
// 权重/偏置由 03_FPGA交接/导出FPGA参数.py 生成的 .mem 文件通过
// $readmemh 装载，数值即两两互补编码的十六进制，位宽与文件一致。
//
// 时序：每个池化输出点 = 4 个卷积子位置（2x2）各做 9 抽头串行 MAC，
// 逐点求 ReLU/量化后取最大值，再写入 flatten_buffer，无需缓存整张
// 26x26 特征图，节省存储。
//
module conv_pool_unit #(
    parameter CONV_WEIGHT_MEMFILE = "../../B_模型交接_v1/generated_fpga/conv_weight_int8.mem",
    parameter CONV_BIAS_MEMFILE   = "../../B_模型交接_v1/generated_fpga/conv_bias_int32.mem"
)(
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,     // 图片已装载完毕，拉高一拍启动
    output reg          busy,
    output reg          done,      // 676 个展平结果写完，拉高一拍

    // 读输入图片缓存（组合地址）
    output wire [4:0]  img_row,
    output wire [4:0]  img_col,
    input  wire [7:0]  img_pixel,

    // 写展平缓存
    output reg          flat_we,
    output reg  [9:0]   flat_waddr,
    output reg  [7:0]   flat_wdata
);

    // ---------------- 参数 ROM ----------------
    // conv_weight_int8: [oc,ic=1,kr,kc] -> 下标 = oc*9 + kr*3 + kc，共 36 项
    reg signed [7:0]  conv_w_mem [0:35];
    // conv_bias_int32: [oc]，共 4 项
    reg signed [31:0] conv_b_mem [0:3];

    initial $readmemh(CONV_WEIGHT_MEMFILE, conv_w_mem);
    initial $readmemh(CONV_BIAS_MEMFILE,   conv_b_mem);

    // ---------------- 计数器 ----------------
    reg [1:0] oc;      // 输出通道 0..3
    reg [3:0] prow;    // 池化输出行 0..12
    reg [3:0] pcol;    // 池化输出列 0..12
    reg [1:0] sub;     // 2x2 子位置 {dy,dx}，0..3
    reg [1:0] kr;      // 卷积核行 0..2
    reg [1:0] kc;      // 卷积核列 0..2

    reg signed [31:0] acc;       // 当前子位置的 9 项乘加和
    reg        [7:0]  pool_max;  // 当前池化窗口内已见过的最大量化值

    wire dy = sub[1];
    wire dx = sub[0];

    // 卷积输出坐标（子位置对应的 26x26 特征图坐标）
    wire [4:0] conv_row = ({prow, 1'b0} + dy);
    wire [4:0] conv_col = ({pcol, 1'b0} + dx);

    // 当前抽头对应的输入像素坐标（组合寻址，input_buffer 异步读）
    assign img_row = conv_row + kr;
    assign img_col = conv_col + kc;

    wire signed [7:0]  cur_weight = conv_w_mem[oc*9 + kr*3 + kc];
    wire signed [31:0] cur_bias   = conv_b_mem[oc];

    // uint8 像素 -> 9 位无符号扩展再当有符号数乘，避免符号误判
    wire signed [16:0] product = $signed({1'b0, img_pixel}) * cur_weight;

    // ReLU + 舍入右移 8 位 + 饱和到 [0,255]
    wire signed [31:0] biased   = acc + cur_bias;
    wire signed [31:0] positive = biased[31] ? 32'sd0 : biased;
    wire        [31:0] rounded  = (positive + 32'd128) >> 8;
    wire        [7:0]  requant  = (rounded > 32'd255) ? 8'd255 : rounded[7:0];

    localparam S_IDLE         = 3'd0,
               S_MAC           = 3'd1,
               S_FINISH_SUB    = 3'd2,
               S_FINISH_BLOCK  = 3'd3,
               S_DONE          = 3'd4;

    reg [2:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            busy       <= 1'b0;
            done       <= 1'b0;
            flat_we    <= 1'b0;
            flat_waddr <= 10'd0;
            flat_wdata <= 8'd0;
            oc <= 2'd0; prow <= 4'd0; pcol <= 4'd0;
            sub <= 2'd0; kr <= 2'd0; kc <= 2'd0;
            acc <= 32'sd0; pool_max <= 8'd0;
        end else begin
            done    <= 1'b0;   // 默认不置位，仅在 S_DONE 拉高一拍
            flat_we <= 1'b0;   // 默认不写，仅在 S_FINISH_BLOCK 拉高一拍

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy <= 1'b1;
                        oc <= 2'd0; prow <= 4'd0; pcol <= 4'd0;
                        sub <= 2'd0; kr <= 2'd0; kc <= 2'd0;
                        acc <= 32'sd0; pool_max <= 8'd0;
                        state <= S_MAC;
                    end
                end

                // 每拍完成一次乘加，走完 3x3=9 个抽头
                S_MAC: begin
                    acc <= acc + product;
                    if (kc == 2'd2) begin
                        kc <= 2'd0;
                        if (kr == 2'd2) begin
                            kr    <= 2'd0;
                            state <= S_FINISH_SUB;
                        end else begin
                            kr <= kr + 2'd1;
                        end
                    end else begin
                        kc <= kc + 2'd1;
                    end
                end

                // 9 项乘加和已在 acc 中：加偏置、ReLU、右移量化、更新池化最大值
                S_FINISH_SUB: begin
                    if (requant > pool_max) pool_max <= requant;
                    acc <= 32'sd0;
                    if (sub == 2'd3) begin
                        sub   <= 2'd0;
                        state <= S_FINISH_BLOCK;
                    end else begin
                        sub   <= sub + 2'd1;
                        state <= S_MAC;
                    end
                end

                // 2x2 池化窗口的最大值写入展平缓存，推进到下一个池化输出点
                S_FINISH_BLOCK: begin
                    flat_we    <= 1'b1;
                    flat_waddr <= oc * 10'd169 + prow * 10'd13 + pcol;
                    flat_wdata <= pool_max;
                    pool_max   <= 8'd0;

                    if (pcol == 4'd12) begin
                        pcol <= 4'd0;
                        if (prow == 4'd12) begin
                            prow <= 4'd0;
                            if (oc == 2'd3) begin
                                state <= S_DONE;
                            end else begin
                                oc    <= oc + 2'd1;
                                state <= S_MAC;
                            end
                        end else begin
                            prow  <= prow + 4'd1;
                            state <= S_MAC;
                        end
                    end else begin
                        pcol  <= pcol + 4'd1;
                        state <= S_MAC;
                    end
                end

                S_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
