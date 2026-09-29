// =============================================================================
// cnn_axil_regs.v —— AXI4-Lite 从口寄存器（地址见 cnn_defines.vh）
// PYNQ 中用 MMIO / overlay.cnn_accel_0.read()/write() 访问
// =============================================================================
`timescale 1ns / 1ps
`include "cnn_defines.vh"

module cnn_axil_regs #(
    parameter ADDR_W = `CNN_AXIL_ADDR_W
)(
    input  wire              aclk,
    input  wire              aresetn,

    input  wire [ADDR_W-1:0] s_axi_awaddr,
    input  wire [2:0]        s_axi_awprot,
    input  wire              s_axi_awvalid,
    output reg               s_axi_awready,
    input  wire [31:0]       s_axi_wdata,
    input  wire [3:0]        s_axi_wstrb,
    input  wire              s_axi_wvalid,
    output reg               s_axi_wready,
    output wire [1:0]        s_axi_bresp,
    output reg               s_axi_bvalid,
    input  wire              s_axi_bready,
    input  wire [ADDR_W-1:0] s_axi_araddr,
    input  wire [2:0]        s_axi_arprot,
    input  wire              s_axi_arvalid,
    output reg               s_axi_arready,
    output reg  [31:0]       s_axi_rdata,
    output wire [1:0]        s_axi_rresp,
    output reg               s_axi_rvalid,
    input  wire              s_axi_rready,

    // 与计算核心的连接
    output reg               soft_rst,
    output wire              irq,
    input  wire              busy,
    input  wire              idle_wait_input,
    input  wire              done_pulse,
    input  wire              err_tlast_pulse,
    input  wire [3:0]        result_class,
    input  wire [31:0]       cycles_last,
    input  wire [31:0]       frame_count,
    input  wire [32*10-1:0]  scores_flat
);

    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;

    reg irq_en, done_st, err_st;
    assign irq = irq_en & done_st;

    // ------------------------------- 写通道 -------------------------------
    wire [ADDR_W-1:0] waddr = s_axi_awaddr;
    reg  clr_done, clr_err;
    always @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_awready <= 1'b0; s_axi_wready <= 1'b0; s_axi_bvalid <= 1'b0;
            soft_rst <= 1'b0; irq_en <= 1'b0; clr_done <= 1'b0; clr_err <= 1'b0;
        end else begin
            soft_rst <= 1'b0; clr_done <= 1'b0; clr_err <= 1'b0;
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;

            if (!s_axi_awready && s_axi_awvalid && s_axi_wvalid && !s_axi_bvalid) begin
                s_axi_awready <= 1'b1;
                s_axi_wready  <= 1'b1;
            end else if (s_axi_awready) begin
                // 本拍 AW/W 握手完成
                s_axi_awready <= 1'b0;
                s_axi_wready  <= 1'b0;
                s_axi_bvalid  <= 1'b1;
                if (s_axi_wstrb[0]) begin
                    case (waddr)
                    `CNN_REG_CTRL:   begin soft_rst <= s_axi_wdata[0]; irq_en <= s_axi_wdata[1]; end
                    `CNN_REG_STATUS: begin clr_done <= s_axi_wdata[1]; clr_err <= s_axi_wdata[2]; end
                    default: ;
                    endcase
                end
            end
        end
    end

    // 粘滞状态位
    always @(posedge aclk) begin
        if (!aresetn) begin
            done_st <= 1'b0; err_st <= 1'b0;
        end else begin
            if (done_pulse)           done_st <= 1'b1;
            else if (clr_done || soft_rst) done_st <= 1'b0;
            if (err_tlast_pulse)      err_st  <= 1'b1;
            else if (clr_err || soft_rst)  err_st  <= 1'b0;
        end
    end

    // ------------------------------- 读通道 -------------------------------
    reg [31:0] rmux;
    always @* begin
        case (s_axi_araddr)
        `CNN_REG_CTRL:    rmux = {30'd0, irq_en, 1'b0};
        `CNN_REG_STATUS:  rmux = {28'd0, idle_wait_input, err_st, done_st, busy};
        `CNN_REG_RESULT:  rmux = {28'd0, result_class};
        `CNN_REG_CYCLES:  rmux = cycles_last;
        `CNN_REG_FRAMES:  rmux = frame_count;
        `CNN_REG_ID:      rmux = `CNN_IP_ID;
        default: begin
            if (s_axi_araddr >= `CNN_REG_SCORE0 && s_axi_araddr < `CNN_REG_SCORE0 + 8'd40)
                rmux = scores_flat[((s_axi_araddr - `CNN_REG_SCORE0) >> 2) * 32 +: 32];
            else
                rmux = 32'hDEAD_BEEF;
        end
        endcase
    end

    always @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_arready <= 1'b0; s_axi_rvalid <= 1'b0; s_axi_rdata <= 32'd0;
        end else begin
            if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 1'b0;
            if (!s_axi_arready && s_axi_arvalid && !s_axi_rvalid) begin
                s_axi_arready <= 1'b1;
            end else if (s_axi_arready) begin
                s_axi_arready <= 1'b0;
                s_axi_rvalid  <= 1'b1;
                s_axi_rdata   <= rmux;
            end
        end
    end

endmodule
