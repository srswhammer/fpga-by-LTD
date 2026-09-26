`timescale 1ns/1ps
//
// fc_argmax_unit：全连接层（676 -> 10）+ argmax。
// scores = fc_bias_int32 + sum(flatten_uint8 * fc_weight_int8)，
// 不做额外右移/饱和（与 integer_inference.py 第 5 步一致）。
// 并列最大分数时取较小类别编号：按类别 0..9 顺序比较，仅用严格 ">"
// 更新最优解，天然保留先出现（编号更小）的那个。
//
module fc_argmax_unit #(
    parameter FC_WEIGHT_MEMFILE = "../../B_模型交接_v1/generated_fpga/fc_weight_int8.mem",
    parameter FC_BIAS_MEMFILE   = "../../B_模型交接_v1/generated_fpga/fc_bias_int32.mem"
)(
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    output reg          busy,
    output reg          done,

    // 读展平缓存（组合地址）
    output wire [9:0]  flat_raddr,
    input  wire [7:0]  flat_rdata,

    // 十个 int32 分数打包输出：scores_packed[32*i +: 32] 为第 i 类分数
    output reg  signed [319:0] scores_packed,
    output reg  [3:0]          digit   // 预测数字 0..9
);

    // fc_weight_int8: [cls,idx] -> 下标 = cls*676+idx，共 6760 项
    reg signed [7:0]  fc_w_mem [0:6759];
    // fc_bias_int32: [cls]，共 10 项
    reg signed [31:0] fc_b_mem [0:9];

    initial $readmemh(FC_WEIGHT_MEMFILE, fc_w_mem);
    initial $readmemh(FC_BIAS_MEMFILE,   fc_b_mem);

    reg  [3:0]  cls;    // 类别 0..9
    reg  [9:0]  idx;    // 展平索引 0..675

    reg signed [31:0] acc;
    reg signed [31:0] scores_reg [0:9];
    reg signed [31:0] best_score;
    reg  [3:0]         best_idx;

    assign flat_raddr = idx;

    wire [12:0] fc_addr    = cls * 13'd676 + idx;
    wire signed [7:0]  cur_weight = fc_w_mem[fc_addr];
    wire signed [31:0] cur_bias   = fc_b_mem[cls];

    wire signed [16:0] product = $signed({1'b0, flat_rdata}) * cur_weight;
    wire signed [31:0] biased  = acc + cur_bias;

    localparam S_IDLE         = 2'd0,
               S_MAC           = 2'd1,
               S_FINISH_CLASS  = 2'd2,
               S_DONE          = 2'd3;

    reg [1:0] state;
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            busy <= 1'b0; done <= 1'b0;
            cls <= 4'd0; idx <= 10'd0; acc <= 32'sd0;
            best_score <= 32'sd0; best_idx <= 4'd0; digit <= 4'd0;
            for (i = 0; i < 10; i = i + 1) scores_reg[i] <= 32'sd0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy  <= 1'b1;
                        cls   <= 4'd0;
                        idx   <= 10'd0;
                        acc   <= 32'sd0;
                        state <= S_MAC;
                    end
                end

                // 每拍完成一次乘加，走完 676 个展平元素
                S_MAC: begin
                    acc <= acc + product;
                    if (idx == 10'd675) begin
                        idx   <= 10'd0;
                        state <= S_FINISH_CLASS;
                    end else begin
                        idx <= idx + 10'd1;
                    end
                end

                // acc 已含 676 项乘加和：加偏置得到该类分数，更新 argmax
                S_FINISH_CLASS: begin
                    scores_reg[cls] <= biased;
                    if (cls == 4'd0 || biased > best_score) begin
                        best_score <= biased;
                        best_idx   <= cls;
                    end
                    acc <= 32'sd0;
                    if (cls == 4'd9) begin
                        state <= S_DONE;
                    end else begin
                        cls   <= cls + 4'd1;
                        state <= S_MAC;
                    end
                end

                S_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    digit <= best_idx;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // 打包输出十个分数
    always @(*) begin
        for (i = 0; i < 10; i = i + 1) begin
            scores_packed[32*i +: 32] = scores_reg[i];
        end
    end

endmodule
