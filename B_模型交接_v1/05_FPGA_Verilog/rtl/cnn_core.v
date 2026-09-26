// =============================================================================
// cnn_core.v —— DigitCNN int8 计算核心（与 02_模型核心/integer_inference.py 逐位一致）
//
// 流程（一帧）：
//   S_LOAD : 从 AXIS 接收 784 像素，拼成 28 行 × 224 bit 存入行存储
//   S_CONV : 对每个池化位置 (pr,pc) 取 4×4 像素块，4 个子位置 × 4 通道并行
//            36 个乘法 → 加偏置 → ReLU/(+128)>>8/饱和 → 2×2 取最大 → 写池化缓存
//            每个池化位置 11 拍，共 169×11 = 1859 拍
//   S_FC   : 按展平索引 idx = c*169 + r*13 + col 遍历 676 个激活，10 类并行 MAC，676+1 拍
//   S_ARG  : 顺序比较 10 个分数（有符号，严格大于 → 并列取小编号），10 拍
//   S_OUT  : （可选）经 m_axis 输出 10 个 int32 分数 [+ argmax]
//   100 MHz 下计算部分约 2.5k 拍 ≈ 25 µs / 帧（不含 DMA）
// =============================================================================
`timescale 1ns / 1ps
`include "cnn_defines.vh"

module cnn_core #(
    parameter IN_W = `CNN_S_AXIS_DATA_W
)(
    input  wire                 clk,
    input  wire                 rst_n,          // 低有效，同步
    input  wire                 soft_rst,       // 高有效单拍，同步

    // 像素输入流
    input  wire [IN_W-1:0]      s_axis_tdata,
    input  wire                 s_axis_tvalid,
    output wire                 s_axis_tready,
    input  wire                 s_axis_tlast,

    // 分数输出流
    input  wire                 out_en,         // 1 = 结果经 m_axis 发送；0 = 跳过发送
    output wire [31:0]          m_axis_tdata,
    output wire                 m_axis_tvalid,
    input  wire                 m_axis_tready,
    output wire                 m_axis_tlast,

    // 状态 / 结果
    output wire                 busy,
    output wire                 idle_wait_input,
    output reg                  done_pulse,
    output reg                  err_tlast_pulse,
    output reg  [3:0]           result_class,
    output reg  [31:0]          cycles_last,
    output reg  [31:0]          frame_count,
    output wire [32*10-1:0]     scores_flat     // {score9,...,score0}
);

    localparam PPB     = IN_W / 8;              // 每拍像素数
    localparam ROW_W   = 8 * 28;                // 224
    localparam S_LOAD  = 3'd0,
               S_CONV  = 3'd1,
               S_FC    = 3'd2,
               S_ARG   = 3'd3,
               S_OUT   = 3'd4;

    initial begin
        if ((28 % PPB) != 0 || (IN_W % 8) != 0) begin
            $display("ERROR: CNN_S_AXIS_DATA_W=%0d 不合法，必须为 8/16/32", IN_W);
            $finish;
        end
    end

    wire srst = (!rst_n) || soft_rst;
    reg  [2:0] state;

    // =========================================================================
    // 参数 ROM（$readmemh 初始化，综合后为常量 / BRAM）
    // =========================================================================
    reg [7:0]  conv_w [0:35];     // 索引 c*9 + kr*3 + kc
    reg [31:0] conv_b [0:3];
    reg [79:0] fc_w   [0:675];    // 每行 10 个 int8，类别 k 在 [k*8 +: 8]
    reg [31:0] fc_b   [0:9];
    initial begin
        $readmemh(`CNN_CONV_W_FILE, conv_w);
        $readmemh(`CNN_CONV_B_FILE, conv_b);
        $readmemh(`CNN_FC_W_FILE,   fc_w);
        $readmemh(`CNN_FC_B_FILE,   fc_b);
    end

    // =========================================================================
    // 1. 输入接收：一行 28 像素拼好后整行写入 img_row
    // =========================================================================
    reg  [ROW_W-1:0] img_row [0:27];
    reg  [ROW_W-1:0] row_buf;
    reg  [ROW_W-1:0] row_next;
    reg  [4:0]       ld_row, ld_col;
    reg              in_frame;         // 已收到本帧第一拍

    assign s_axis_tready = (state == S_LOAD);
    wire beat      = s_axis_tvalid && s_axis_tready;
    wire row_end   = (ld_col == 28 - PPB);
    wire last_beat = row_end && (ld_row == 27);

    always @* begin
        row_next = row_buf;
        row_next[ld_col*8 +: IN_W] = s_axis_tdata;
    end

    // =========================================================================
    // 2. 卷积 + ReLU + 池化
    // =========================================================================
    reg  [3:0]  pr, pc;                // 池化输出行/列 0..12
    reg  [3:0]  step;                  // 每个池化位置 0..10
    wire [4:0]  rd_addr = {pr, 1'b0} + {3'd0, step[1:0]};
    reg  [ROW_W-1:0] img_q;
    reg  [31:0] patch [0:3];           // 4×4 像素块，每行 4 字节，低字节为左列
    wire [7:0]  pos = pr * 8'd13 + pc; // 池化平面地址 0..168

    // 行存储：同步写 / 同步读（可推断为 RAM）
    always @(posedge clk) begin
        if (beat && row_end) img_row[ld_row] <= row_next;
        img_q <= img_row[rd_addr];
    end

    // 取 4×4 块：step1..4 依次得到第 2pr+0..3 行
    always @(posedge clk) begin
        if (state == S_CONV && step >= 4'd1 && step <= 4'd4)
            patch[step[1:0] - 2'd1] <= img_q[pc*16 +: 32];
    end

    // step5..8 依次发射 2×2 子位置 sp=0..3 （dy=sp[1], dx=sp[0]）
    wire       issue = (state == S_CONV) && (step >= 4'd5) && (step <= 4'd8);
    wire [1:0] sp    = step[1:0] - 2'd1;       // 5→0, 6→1, 7→2, 8→3
    wire       dy    = sp[1];
    wire       dx    = sp[0];

    reg [7:0]  px [0:8];                       // 当前 3×3 窗口
    reg [31:0] prow;
    integer kr, kc;
    always @* begin
        for (kr = 0; kr < 3; kr = kr + 1) begin
            prow = patch[dy + kr];
            for (kc = 0; kc < 3; kc = kc + 1)
                px[kr*3 + kc] = prow[(dx + kc)*8 +: 8];
        end
    end

    // 流水 1：36 个 uint8×int8 乘法
    reg signed [16:0] prod [0:35];
    reg v1, first1, last1;
    integer i1;
    always @(posedge clk) begin
        for (i1 = 0; i1 < 36; i1 = i1 + 1)
            prod[i1] <= $signed({1'b0, px[i1 % 9]}) * $signed(conv_w[i1]);
    end

    // 流水 2：9 项求和 + 偏置 → int32 累加值
    reg signed [31:0] acc [0:3];
    reg signed [31:0] acc_next [0:3];
    reg v2, first2, last2;
    integer c2, k2;
    always @* begin
        for (c2 = 0; c2 < 4; c2 = c2 + 1) begin
            acc_next[c2] = $signed(conv_b[c2]);
            for (k2 = 0; k2 < 9; k2 = k2 + 1)
                acc_next[c2] = acc_next[c2] + prod[c2*9 + k2];
        end
    end
    always @(posedge clk) begin
        for (c2 = 0; c2 < 4; c2 = c2 + 1) acc[c2] <= acc_next[c2];
    end

    always @(posedge clk) begin
        if (srst) begin
            v1 <= 1'b0; v2 <= 1'b0;
            first1 <= 1'b0; last1 <= 1'b0; first2 <= 1'b0; last2 <= 1'b0;
        end else begin
            v1 <= issue;       first1 <= issue && (sp == 2'd0); last1 <= issue && (sp == 2'd3);
            v2 <= v1;          first2 <= first1;                last2 <= last1;
        end
    end

    // ReLU + 四舍五入右移 + 饱和：sat_u8((max(acc,0) + 128) >> 8)
    function [7:0] relu_req;
        input signed [31:0] a;
        reg [32:0] t;
        begin
            if (a[31]) relu_req = 8'd0;
            else begin
                t = ({1'b0, a} + (33'd1 << (`CNN_REQ_SHIFT - 1))) >> `CNN_REQ_SHIFT;
                relu_req = (t > 33'd255) ? 8'd255 : t[7:0];
            end
        end
    endfunction

    reg  [7:0] mx   [0:3];
    reg  [7:0] mnew [0:3];
    reg  [7:0] rq;
    integer c3;
    always @* begin
        for (c3 = 0; c3 < 4; c3 = c3 + 1) begin
            rq = relu_req(acc[c3]);
            mnew[c3] = (first2 || rq > mx[c3]) ? rq : mx[c3];
        end
    end

    // 池化缓存：4 个通道各 169 字节，同拍写入
    reg [7:0] pool_b0 [0:168];
    reg [7:0] pool_b1 [0:168];
    reg [7:0] pool_b2 [0:168];
    reg [7:0] pool_b3 [0:168];
    wire pool_we = v2 && last2;

    always @(posedge clk) begin
        if (v2) begin
            mx[0] <= mnew[0]; mx[1] <= mnew[1]; mx[2] <= mnew[2]; mx[3] <= mnew[3];
        end
        if (pool_we) begin
            pool_b0[pos] <= mnew[0];
            pool_b1[pos] <= mnew[1];
            pool_b2[pos] <= mnew[2];
            pool_b3[pos] <= mnew[3];
        end
    end

    // =========================================================================
    // 3. 全连接：每拍 1 个激活 × 10 类
    // =========================================================================
    reg  [1:0]  fc_c, fc_c_d;
    reg  [7:0]  fc_pos;
    reg  [9:0]  fc_idx;
    reg         fc_run, fc_v;
    reg  [79:0] fc_w_q;
    reg  [7:0]  pq0, pq1, pq2, pq3;
    reg  signed [31:0] facc [0:9];

    always @(posedge clk) begin
        fc_w_q <= fc_w[fc_idx];
        pq0 <= pool_b0[fc_pos];
        pq1 <= pool_b1[fc_pos];
        pq2 <= pool_b2[fc_pos];
        pq3 <= pool_b3[fc_pos];
    end
    wire [7:0] fx = (fc_c_d == 2'd0) ? pq0 :
                    (fc_c_d == 2'd1) ? pq1 :
                    (fc_c_d == 2'd2) ? pq2 : pq3;

    // =========================================================================
    // 4. argmax / 输出
    // =========================================================================
    reg  [3:0]  arg_i, best_i;
    reg  signed [31:0] best_v;
    reg  [31:0] score_o [0:9];          // 锁存的最终分数（AXI-Lite 读取稳定）
    reg  [3:0]  out_i;
    reg  [31:0] cyc;

    genvar gs;
    generate for (gs = 0; gs < 10; gs = gs + 1) begin : g_sc
        assign scores_flat[gs*32 +: 32] = score_o[gs];
    end endgenerate

    assign m_axis_tvalid = (state == S_OUT);
    assign m_axis_tdata  = (out_i < 4'd10) ? score_o[out_i] : {28'd0, result_class};
    assign m_axis_tlast  = (out_i == `CNN_OUT_WORDS - 1);

    assign busy            = (state != S_LOAD) || in_frame;
    assign idle_wait_input = (state == S_LOAD) && !in_frame;

    // =========================================================================
    // 主状态机
    // =========================================================================
    integer j;
    always @(posedge clk) begin
        if (srst) begin
            state <= S_LOAD;
            ld_row <= 5'd0; ld_col <= 5'd0; row_buf <= {ROW_W{1'b0}}; in_frame <= 1'b0;
            pr <= 4'd0; pc <= 4'd0; step <= 4'd0;
            fc_c <= 2'd0; fc_c_d <= 2'd0; fc_pos <= 8'd0; fc_idx <= 10'd0; fc_run <= 1'b0; fc_v <= 1'b0;
            arg_i <= 4'd0; best_i <= 4'd0; best_v <= 32'sd0; out_i <= 4'd0;
            result_class <= 4'd0; cycles_last <= 32'd0; frame_count <= 32'd0; cyc <= 32'd0;
            done_pulse <= 1'b0; err_tlast_pulse <= 1'b0;
            for (j = 0; j < 10; j = j + 1) score_o[j] <= 32'd0;
        end else begin
            done_pulse      <= 1'b0;
            err_tlast_pulse <= 1'b0;
            if (in_frame || state != S_LOAD) cyc <= cyc + 32'd1;

            case (state)
            // ---------------------------------------------------------------
            S_LOAD: if (beat) begin
                if (!in_frame) cyc <= 32'd1;
                in_frame <= 1'b1;
                row_buf  <= row_next;
                if (s_axis_tlast && !last_beat) begin
                    // TLAST 提前：丢弃本帧，回到等待
                    err_tlast_pulse <= 1'b1;
                    ld_row <= 5'd0; ld_col <= 5'd0; in_frame <= 1'b0;
                end else if (row_end) begin
                    ld_col <= 5'd0;
                    if (last_beat) begin
                        if (!s_axis_tlast) err_tlast_pulse <= 1'b1;  // 缺 TLAST：报错但照常计算
                        ld_row <= 5'd0; in_frame <= 1'b0;
                        pr <= 4'd0; pc <= 4'd0; step <= 4'd0;
                        state <= S_CONV;
                    end else
                        ld_row <= ld_row + 5'd1;
                end else
                    ld_col <= ld_col + PPB[4:0];
            end
            // ---------------------------------------------------------------
            S_CONV: begin
                if (step == 4'd10) begin
                    step <= 4'd0;
                    if (pc == 4'd12) begin
                        pc <= 4'd0;
                        if (pr == 4'd12) begin
                            pr <= 4'd0;
                            state  <= S_FC;
                            fc_c   <= 2'd0; fc_pos <= 8'd0; fc_idx <= 10'd0;
                            fc_run <= 1'b1;
                            for (j = 0; j < 10; j = j + 1) facc[j] <= $signed(fc_b[j]);
                        end else
                            pr <= pr + 4'd1;
                    end else
                        pc <= pc + 4'd1;
                end else
                    step <= step + 4'd1;
            end
            // ---------------------------------------------------------------
            S_FC: begin
                fc_v   <= fc_run;
                fc_c_d <= fc_c;
                if (fc_run) begin
                    fc_idx <= fc_idx + 10'd1;
                    if (fc_pos == 8'd168) begin
                        fc_pos <= 8'd0;
                        if (fc_c == 2'd3) fc_run <= 1'b0;
                        else              fc_c   <= fc_c + 2'd1;
                    end else
                        fc_pos <= fc_pos + 8'd1;
                end
                if (fc_v)
                    for (j = 0; j < 10; j = j + 1)
                        facc[j] <= facc[j] + $signed({1'b0, fx}) * $signed(fc_w_q[j*8 +: 8]);
                if (!fc_run && !fc_v && fc_idx != 10'd0) begin
                    state  <= S_ARG;
                    fc_idx <= 10'd0;
                    arg_i  <= 4'd1; best_i <= 4'd0; best_v <= facc[0];
                end
            end
            // ---------------------------------------------------------------
            S_ARG: begin
                if (arg_i == 4'd10) begin
                    result_class <= best_i;
                    for (j = 0; j < 10; j = j + 1) score_o[j] <= facc[j];
                    cycles_last  <= cyc;
                    frame_count  <= frame_count + 32'd1;
                    done_pulse   <= 1'b1;
                    out_i        <= 4'd0;
                    state        <= out_en ? S_OUT : S_LOAD;
                end else begin
                    if (facc[arg_i] > best_v) begin    // 严格大于：并列取较小编号
                        best_v <= facc[arg_i];
                        best_i <= arg_i;
                    end
                    arg_i <= arg_i + 4'd1;
                end
            end
            // ---------------------------------------------------------------
            S_OUT: if (m_axis_tready) begin
                if (out_i == `CNN_OUT_WORDS - 1) state <= S_LOAD;
                else                             out_i <= out_i + 4'd1;
            end
            default: state <= S_LOAD;
            endcase
        end
    end

endmodule
