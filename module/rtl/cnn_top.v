`timescale 1ns/1ps
//
// cnn_top：整数手写数字识别 CNN 加速器顶层。
//
// 结构对齐 B_模型交接_v1/02_模型核心/integer_inference.py：
//   uint8[28,28] -> Conv(1->4,3x3) -> ReLU/右移8位/饱和 -> MaxPool(2x2)
//                -> Flatten(676)   -> FC(676->10) -> argmax
//
// 当前接口是最简单的寄存器/存储器式握手（写图片字节 + start/busy/done +
// 分数输出），尚未绑定 AXI-Lite/AXI-Stream，因为
// B_模型交接_v1/03_FPGA交接/接口说明.md 里这部分协议还未与 C 确认。
// 后续接 DMA/AXI 时，在这一层外面再包一层适配壳即可，无需改动内部数据通路。
//
module cnn_top #(
    parameter CONV_WEIGHT_MEMFILE = "../../B_模型交接_v1/generated_fpga/conv_weight_int8.mem",
    parameter CONV_BIAS_MEMFILE   = "../../B_模型交接_v1/generated_fpga/conv_bias_int32.mem",
    parameter FC_WEIGHT_MEMFILE   = "../../B_模型交接_v1/generated_fpga/fc_weight_int8.mem",
    parameter FC_BIAS_MEMFILE     = "../../B_模型交接_v1/generated_fpga/fc_bias_int32.mem"
)(
    input  wire        clk,
    input  wire        rst_n,

    // 装载 784 字节输入图片：行主序，列变化最快（与预处理输出一致）
    input  wire         img_wr_en,
    input  wire [9:0]   img_wr_addr,   // 0..783
    input  wire [7:0]   img_wr_data,

    // 控制：图片装载完成后拉高一拍 start；busy 期间不可再装载/再启动
    input  wire         start,
    output wire         busy,
    output wire         done,          // 推理完成，拉高一拍

    // 结果：十个有符号 int32 分数（打包）+ 预测数字（并列取较小编号）
    output wire signed [319:0] scores_packed,
    output wire [3:0]          digit
);

    // ---------------- 输入图片缓存 ----------------
    wire [4:0] img_row;
    wire [4:0] img_col;
    wire [7:0] img_pixel;

    input_buffer u_input_buffer (
        .clk   (clk),
        .we    (img_wr_en),
        .waddr (img_wr_addr),
        .wdata (img_wr_data),
        .rrow  (img_row),
        .rcol  (img_col),
        .rdata (img_pixel)
    );

    // ---------------- 展平缓存（卷积+ReLU+池化结果） ----------------
    wire       flat_we;
    wire [9:0] flat_waddr;
    wire [7:0] flat_wdata;
    wire [9:0] flat_raddr;
    wire [7:0] flat_rdata;

    flatten_buffer u_flatten_buffer (
        .clk   (clk),
        .we    (flat_we),
        .waddr (flat_waddr),
        .wdata (flat_wdata),
        .raddr (flat_raddr),
        .rdata (flat_rdata)
    );

    // ---------------- 卷积 + ReLU + 池化 ----------------
    reg  conv_start;
    wire conv_busy, conv_done;

    conv_pool_unit #(
        .CONV_WEIGHT_MEMFILE (CONV_WEIGHT_MEMFILE),
        .CONV_BIAS_MEMFILE   (CONV_BIAS_MEMFILE)
    ) u_conv_pool (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (conv_start),
        .busy       (conv_busy),
        .done       (conv_done),
        .img_row    (img_row),
        .img_col    (img_col),
        .img_pixel  (img_pixel),
        .flat_we    (flat_we),
        .flat_waddr (flat_waddr),
        .flat_wdata (flat_wdata)
    );

    // ---------------- 全连接 + argmax ----------------
    reg  fc_start;
    wire fc_busy, fc_done;

    fc_argmax_unit #(
        .FC_WEIGHT_MEMFILE (FC_WEIGHT_MEMFILE),
        .FC_BIAS_MEMFILE   (FC_BIAS_MEMFILE)
    ) u_fc (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (fc_start),
        .busy           (fc_busy),
        .done           (fc_done),
        .flat_raddr     (flat_raddr),
        .flat_rdata     (flat_rdata),
        .scores_packed  (scores_packed),
        .digit          (digit)
    );

    // ---------------- 顶层顺序控制：先卷积池化，完成后启动全连接 ----------------
    localparam T_IDLE = 2'd0,
               T_CONV = 2'd1,
               T_FC   = 2'd2,
               T_DONE = 2'd3;

    reg [1:0] tstate;
    reg       done_r;

    assign busy = (tstate != T_IDLE);
    assign done = done_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tstate     <= T_IDLE;
            conv_start <= 1'b0;
            fc_start   <= 1'b0;
            done_r     <= 1'b0;
        end else begin
            conv_start <= 1'b0;
            fc_start   <= 1'b0;
            done_r     <= 1'b0;

            case (tstate)
                T_IDLE: begin
                    if (start) begin
                        conv_start <= 1'b1;
                        tstate     <= T_CONV;
                    end
                end
                T_CONV: begin
                    if (conv_done) begin
                        fc_start <= 1'b1;
                        tstate   <= T_FC;
                    end
                end
                T_FC: begin
                    if (fc_done) begin
                        done_r <= 1'b1;
                        tstate <= T_DONE;
                    end
                end
                T_DONE: begin
                    tstate <= T_IDLE;
                end
                default: tstate <= T_IDLE;
            endcase
        end
    end

endmodule
