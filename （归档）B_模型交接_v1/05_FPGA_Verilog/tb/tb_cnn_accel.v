// =============================================================================
// tb_cnn_accel.v —— 用 hardware_reference.npz 的固定十张图逐层比对
//   1) 池化/展平结果 676 字节  2) 十类 int32 分数  3) argmax
//   同时随机 TVALID 空拍 + 随机 TREADY 反压，并测试一次 TLAST 提前的错误帧
// 运行：见 scripts/run_sim.sh（iverilog）或在 Vivado xsim 中将本文件设为 sim top
// =============================================================================
`timescale 1ns / 1ps
`include "cnn_defines.vh"

module tb_cnn_accel;
    localparam IN_W = `CNN_S_AXIS_DATA_W;
    localparam PPB  = IN_W / 8;
    localparam NIMG = 10;

    reg aclk = 0, aresetn = 0;
    always #5 aclk = ~aclk;   // 100 MHz

    reg  [IN_W-1:0] s_tdata;  reg s_tvalid, s_tlast;  wire s_tready;
    wire [31:0] m_tdata; wire m_tvalid, m_tlast; reg m_tready;
    wire [3:0]  m_tkeep;
    reg  [`CNN_AXIL_ADDR_W-1:0] awaddr, araddr;
    reg  awvalid, wvalid, bready, arvalid, rready;
    reg  [31:0] wdata;
    wire awready, wready, bvalid, arready, rvalid;
    wire [1:0] bresp, rresp;
    wire [31:0] rdata;
    wire irq;

    cnn_accel_top dut (
        .aclk(aclk), .aresetn(aresetn),
        .s_axis_tdata(s_tdata), .s_axis_tkeep({PPB{1'b1}}), .s_axis_tvalid(s_tvalid),
        .s_axis_tready(s_tready), .s_axis_tlast(s_tlast)
`ifdef CNN_USE_M_AXIS
       ,.m_axis_tdata(m_tdata), .m_axis_tkeep(m_tkeep), .m_axis_tvalid(m_tvalid),
        .m_axis_tready(m_tready), .m_axis_tlast(m_tlast)
`endif
`ifdef CNN_USE_AXI_LITE
       ,.s_axi_awaddr(awaddr), .s_axi_awprot(3'd0), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arprot(3'd0), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready)
`endif
`ifdef CNN_USE_IRQ
       ,.interrupt(irq)
`endif
    );

    reg [7:0]  img   [0:NIMG*784-1];
    reg [7:0]  poolr [0:NIMG*676-1];
    reg [31:0] scr   [0:NIMG*10-1];
    reg [7:0]  pred  [0:NIMG-1];
    initial begin
        $readmemh("vectors/inputs_u8.mem", img);
        $readmemh("vectors/pool_u8.mem",   poolr);
        $readmemh("vectors/scores_i32.mem", scr);
        $readmemh("vectors/pred.mem",      pred);
    end

    integer errors = 0;
    integer seed   = 12345;

    // ---------------- AXI-Lite 任务 ----------------
    task axil_write(input [7:0] a, input [31:0] d);
    begin
        @(posedge aclk); awaddr <= a; wdata <= d; awvalid <= 1; wvalid <= 1; bready <= 1;
        @(posedge aclk); while (!(awready && awvalid)) @(posedge aclk);
        awvalid <= 0; wvalid <= 0;
        while (!bvalid) @(posedge aclk);
        @(posedge aclk); bready <= 0;
    end
    endtask

    task axil_read(input [7:0] a, output [31:0] d);
    begin
        @(posedge aclk); araddr <= a; arvalid <= 1; rready <= 1;
        @(posedge aclk); while (!(arready && arvalid)) @(posedge aclk);
        arvalid <= 0;
        @(posedge aclk); while (!rvalid) @(posedge aclk);
        d = rdata;
        @(posedge aclk); rready <= 0;
    end
    endtask

    // ---------------- 发送一帧 ----------------
    task send_frame(input integer n, input integer early_last_beat);
        integer b, k, nb;
    begin
        nb = 784 / PPB;
        for (b = 0; b < nb; b = b + 1) begin
            while (($random(seed) & 3) == 0) begin  // 随机空拍
                @(posedge aclk); s_tvalid <= 0;
            end
            @(posedge aclk);
            for (k = 0; k < PPB; k = k + 1) s_tdata[k*8 +: 8] <= img[n*784 + b*PPB + k];
            s_tvalid <= 1;
            s_tlast  <= (early_last_beat >= 0) ? (b == early_last_beat) : (b == nb - 1);
            @(negedge aclk); while (!s_tready) @(negedge aclk);
            if (early_last_beat >= 0 && b == early_last_beat) begin
                @(posedge aclk); s_tvalid <= 0; s_tlast <= 0; b = nb;  // 结束
            end
        end
        @(posedge aclk); s_tvalid <= 0; s_tlast <= 0;
    end
    endtask

    // ---------------- 接收输出流 ----------------
    reg [31:0] got [0:15];
    integer    got_n;
    reg        got_last_ok;
    always @(posedge aclk) begin
        m_tready <= ($random(seed) & 1);           // 随机反压
    end
    always @(posedge aclk) begin
        if (m_tvalid && m_tready) begin
            got[got_n] <= m_tdata;
            if (m_tlast) got_last_ok <= (got_n == `CNN_OUT_WORDS - 1);
            got_n <= got_n + 1;
        end
    end

    // ---------------- 主流程 ----------------
    integer n, i, p, t0;
    reg [31:0] rd;
    initial begin
        s_tdata = 0; s_tvalid = 0; s_tlast = 0; m_tready = 0;
        awaddr = 0; araddr = 0; awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0; wdata = 0;
        got_n = 0; got_last_ok = 0;
        repeat (10) @(posedge aclk);
        aresetn = 1;
        repeat (5) @(posedge aclk);

`ifdef CNN_USE_AXI_LITE
        axil_read(`CNN_REG_ID, rd);
        if (rd !== `CNN_IP_ID) begin $display("FAIL ID = %h", rd); errors = errors + 1; end
        else $display("ID 寄存器 = %h  OK", rd);
        axil_write(`CNN_REG_CTRL, 32'h2);          // 打开中断
`endif

        // ---- 错误帧：TLAST 在第 10 拍提前出现，应被丢弃并置错误位 ----
        send_frame(0, 10);
        repeat (20) @(posedge aclk);
`ifdef CNN_USE_AXI_LITE
        axil_read(`CNN_REG_STATUS, rd);
        if (rd[2] !== 1'b1) begin $display("FAIL: TLAST 提前未报错 status=%h", rd); errors = errors + 1; end
        else $display("TLAST 提前错误帧：已报错并丢弃  OK");
        axil_write(`CNN_REG_STATUS, 32'h4);        // W1C 清错误位
`endif

        for (n = 0; n < NIMG; n = n + 1) begin
            got_n = 0; got_last_ok = 0;
            t0 = $time;
            send_frame(n, -1);
            // 等待完成
`ifdef CNN_USE_M_AXIS
            while (got_n < `CNN_OUT_WORDS) @(posedge aclk);
`else
            while (dut.u_core.done_pulse !== 1'b1) @(posedge aclk);
`endif
            repeat (2) @(posedge aclk);

            // 1) 池化 / 展平层比对
            for (i = 0; i < 676; i = i + 1) begin
                p = i % 169;
                case (i / 169)
                    0: rd = dut.u_core.pool_b0[p];
                    1: rd = dut.u_core.pool_b1[p];
                    2: rd = dut.u_core.pool_b2[p];
                    default: rd = dut.u_core.pool_b3[p];
                endcase
                if (rd[7:0] !== poolr[n*676 + i]) begin
                    if (errors < 20) $display("FAIL img%0d flatten[%0d] = %0d, 期望 %0d", n, i, rd[7:0], poolr[n*676+i]);
                    errors = errors + 1;
                end
            end
`ifdef CNN_USE_M_AXIS
            // 2) 分数比对（输出流）
            for (i = 0; i < 10; i = i + 1)
                if (got[i] !== scr[n*10 + i]) begin
                    $display("FAIL img%0d score[%0d] = %0d, 期望 %0d", n, i, $signed(got[i]), $signed(scr[n*10+i]));
                    errors = errors + 1;
                end
            if (!got_last_ok) begin $display("FAIL img%0d TLAST 位置错误", n); errors = errors + 1; end
  `ifdef CNN_OUT_ARGMAX_WORD
            if (got[10] !== pred[n]) begin $display("FAIL img%0d 流中 argmax=%0d 期望 %0d", n, got[10], pred[n]); errors = errors + 1; end
  `endif
`endif
`ifdef CNN_USE_AXI_LITE
            // 3) AXI-Lite 读结果 / 分数
            axil_read(`CNN_REG_RESULT, rd);
            if (rd[3:0] !== pred[n][3:0]) begin $display("FAIL img%0d RESULT=%0d 期望 %0d", n, rd, pred[n]); errors = errors + 1; end
            for (i = 0; i < 10; i = i + 1) begin
                axil_read(`CNN_REG_SCORE0 + 4*i, rd);
                if (rd !== scr[n*10 + i]) begin $display("FAIL img%0d REG score[%0d]=%0d", n, i, $signed(rd)); errors = errors + 1; end
            end
            axil_read(`CNN_REG_CYCLES, rd);
            $display("图 %0d：预测 %0d（期望 %0d），核心周期 %0d，score[pred]=%0d",
                     n, dut.u_core.result_class, pred[n], rd, $signed(dut.u_core.score_o[dut.u_core.result_class]));
            if (irq !== 1'b1) begin $display("FAIL img%0d 中断未拉高", n); errors = errors + 1; end
            axil_write(`CNN_REG_STATUS, 32'h2);    // 清 done → 中断撤销
`else
            $display("图 %0d：预测 %0d（期望 %0d）", n, dut.u_core.result_class, pred[n]);
`endif
        end

`ifdef CNN_USE_AXI_LITE
        axil_read(`CNN_REG_FRAMES, rd);
        if (rd !== NIMG) begin $display("FAIL FRAMES=%0d", rd); errors = errors + 1; end
`endif

        if (errors == 0) $display("\n==== PASS：十张图 池化层 / 十类分数 / argmax 与 CPU 整数标准答案逐位一致 ====");
        else             $display("\n==== FAIL：共 %0d 处不一致 ====", errors);
        $finish;
    end

    initial begin
        #20_000_000;
        $display("TIMEOUT"); $finish;
    end
endmodule
