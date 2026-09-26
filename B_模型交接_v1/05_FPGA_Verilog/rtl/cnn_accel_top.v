// =============================================================================
// cnn_accel_top.v —— 顶层（Vivado 打包 IP / 作为 RTL 模块拖入 Block Design）
//
// 端口命名遵循 Vivado 自动识别规则：
//   aclk / aresetn           → 时钟复位（接 FCLK_CLK0 + proc_sys_reset peripheral_aresetn）
//   s_axis_*                 → AXIS 从口（接 axi_dma_0/M_AXIS_MM2S）
//   m_axis_*                 → AXIS 主口（接 axi_dma_0/S_AXIS_S2MM）
//   s_axi_*                  → AXI4-Lite 从口（接 ps7 M_AXI_GP0 → interconnect）
//   interrupt                → 完成中断（高电平，接 xlconcat → IRQ_F2P）
// 哪些端口存在由 cnn_defines.vh 中的宏决定。
// =============================================================================
`timescale 1ns / 1ps
`include "cnn_defines.vh"

module cnn_accel_top (
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis:s_axi, ASSOCIATED_RESET aresetn" *)
    input  wire                              aclk,
    input  wire                              aresetn,

    // ---------------- AXIS 像素输入 ----------------
    input  wire [`CNN_S_AXIS_DATA_W-1:0]     s_axis_tdata,
    input  wire [`CNN_S_AXIS_DATA_W/8-1:0]   s_axis_tkeep,    // 未使用：要求整拍有效
    input  wire                              s_axis_tvalid,
    output wire                              s_axis_tready,
    input  wire                              s_axis_tlast

`ifdef CNN_USE_M_AXIS
    // ---------------- AXIS 分数输出 ----------------
   ,output wire [`CNN_M_AXIS_DATA_W-1:0]     m_axis_tdata
   ,output wire [`CNN_M_AXIS_DATA_W/8-1:0]   m_axis_tkeep
   ,output wire                              m_axis_tvalid
   ,input  wire                              m_axis_tready
   ,output wire                              m_axis_tlast
`endif

`ifdef CNN_USE_AXI_LITE
    // ---------------- AXI4-Lite 控制 ----------------
   ,input  wire [`CNN_AXIL_ADDR_W-1:0]       s_axi_awaddr
   ,input  wire [2:0]                        s_axi_awprot
   ,input  wire                              s_axi_awvalid
   ,output wire                              s_axi_awready
   ,input  wire [31:0]                       s_axi_wdata
   ,input  wire [3:0]                        s_axi_wstrb
   ,input  wire                              s_axi_wvalid
   ,output wire                              s_axi_wready
   ,output wire [1:0]                        s_axi_bresp
   ,output wire                              s_axi_bvalid
   ,input  wire                              s_axi_bready
   ,input  wire [`CNN_AXIL_ADDR_W-1:0]       s_axi_araddr
   ,input  wire [2:0]                        s_axi_arprot
   ,input  wire                              s_axi_arvalid
   ,output wire                              s_axi_arready
   ,output wire [31:0]                       s_axi_rdata
   ,output wire [1:0]                        s_axi_rresp
   ,output wire                              s_axi_rvalid
   ,input  wire                              s_axi_rready
`endif

`ifdef CNN_USE_IRQ
   ,
    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 interrupt INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "SENSITIVITY LEVEL_HIGH" *)
    output wire                              interrupt
`endif
);

    wire              soft_rst;
    wire              busy, idle_wait_input, done_pulse, err_tlast_pulse;
    wire [3:0]        result_class;
    wire [31:0]       cycles_last, frame_count;
    wire [32*10-1:0]  scores_flat;
    wire              irq_w;

    wire [31:0]       core_m_tdata;
    wire              core_m_tvalid, core_m_tlast;
    wire              core_m_tready;

`ifdef CNN_USE_M_AXIS
    assign m_axis_tdata  = core_m_tdata;
    assign m_axis_tkeep  = {(`CNN_M_AXIS_DATA_W/8){1'b1}};
    assign m_axis_tvalid = core_m_tvalid;
    assign m_axis_tlast  = core_m_tlast;
    assign core_m_tready = m_axis_tready;
    localparam OUT_EN = 1'b1;
`else
    assign core_m_tready = 1'b1;
    localparam OUT_EN = 1'b0;
`endif

    cnn_core #(.IN_W(`CNN_S_AXIS_DATA_W)) u_core (
        .clk             (aclk),
        .rst_n           (aresetn),
        .soft_rst        (soft_rst),
        .s_axis_tdata    (s_axis_tdata),
        .s_axis_tvalid   (s_axis_tvalid),
        .s_axis_tready   (s_axis_tready),
        .s_axis_tlast    (s_axis_tlast),
        .out_en          (OUT_EN),
        .m_axis_tdata    (core_m_tdata),
        .m_axis_tvalid   (core_m_tvalid),
        .m_axis_tready   (core_m_tready),
        .m_axis_tlast    (core_m_tlast),
        .busy            (busy),
        .idle_wait_input (idle_wait_input),
        .done_pulse      (done_pulse),
        .err_tlast_pulse (err_tlast_pulse),
        .result_class    (result_class),
        .cycles_last     (cycles_last),
        .frame_count     (frame_count),
        .scores_flat     (scores_flat)
    );

`ifdef CNN_USE_AXI_LITE
    cnn_axil_regs #(.ADDR_W(`CNN_AXIL_ADDR_W)) u_regs (
        .aclk            (aclk),
        .aresetn         (aresetn),
        .s_axi_awaddr    (s_axi_awaddr),
        .s_axi_awprot    (s_axi_awprot),
        .s_axi_awvalid   (s_axi_awvalid),
        .s_axi_awready   (s_axi_awready),
        .s_axi_wdata     (s_axi_wdata),
        .s_axi_wstrb     (s_axi_wstrb),
        .s_axi_wvalid    (s_axi_wvalid),
        .s_axi_wready    (s_axi_wready),
        .s_axi_bresp     (s_axi_bresp),
        .s_axi_bvalid    (s_axi_bvalid),
        .s_axi_bready    (s_axi_bready),
        .s_axi_araddr    (s_axi_araddr),
        .s_axi_arprot    (s_axi_arprot),
        .s_axi_arvalid   (s_axi_arvalid),
        .s_axi_arready   (s_axi_arready),
        .s_axi_rdata     (s_axi_rdata),
        .s_axi_rresp     (s_axi_rresp),
        .s_axi_rvalid    (s_axi_rvalid),
        .s_axi_rready    (s_axi_rready),
        .soft_rst        (soft_rst),
        .irq             (irq_w),
        .busy            (busy),
        .idle_wait_input (idle_wait_input),
        .done_pulse      (done_pulse),
        .err_tlast_pulse (err_tlast_pulse),
        .result_class    (result_class),
        .cycles_last     (cycles_last),
        .frame_count     (frame_count),
        .scores_flat     (scores_flat)
    );
`else
    assign soft_rst = 1'b0;
    // 无 AXI-Lite 时中断退化为 done 单拍脉冲展宽（简单方案，按需修改）
    reg [3:0] irq_stretch;
    always @(posedge aclk)
        if (!aresetn) irq_stretch <= 4'd0;
        else          irq_stretch <= done_pulse ? 4'hF : {irq_stretch[2:0], 1'b0};
    assign irq_w = irq_stretch[3];
`endif

`ifdef CNN_USE_IRQ
    assign interrupt = irq_w;
`endif

endmodule
