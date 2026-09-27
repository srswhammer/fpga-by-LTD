module camera_diag_top (
    input         sys_clk,
    input         key_n,
    input         cam_pclk,
    input         cam_vsync,
    input         cam_href,
    input  [7:0]  cam_data,
    output        cam_rst_n,
    output        cam_pwdn,
    output        cam_scl,
    inout         cam_sda,
    output        led_init,
    output        led_pclk
);

wire        i2c_dri_clk;
wire        i2c_done;
wire [7:0]  i2c_data_r;
wire        i2c_exec;
wire [23:0] i2c_data;
wire        i2c_rh_wl;
wire        cam_init_done;
reg  [25:0] pclk_count;

assign cam_pwdn  = 1'b0;
assign cam_rst_n = 1'b1;
assign led_init  = cam_init_done;
assign led_pclk  = pclk_count[25];

always @(posedge cam_pclk or negedge key_n) begin
    if (!key_n)
        pclk_count <= 26'd0;
    else
        pclk_count <= pclk_count + 1'b1;
end

i2c_ov5640_rgb565_cfg u_cfg (
    .clk(i2c_dri_clk),
    .rst_n(key_n),
    .i2c_data_r(i2c_data_r),
    .i2c_done(i2c_done),
    .cmos_h_pixel(13'd640),
    .cmos_v_pixel(13'd480),
    .total_h_pixel(13'd1856),
    .total_v_pixel(13'd984),
    .i2c_exec(i2c_exec),
    .i2c_data(i2c_data),
    .i2c_rh_wl(i2c_rh_wl),
    .init_done(cam_init_done)
);

i2c_dri #(
    .SLAVE_ADDR(7'h3c),
    .CLK_FREQ(27'd50_000_000),
    .I2C_FREQ(20'd250_000)
) u_i2c (
    .clk(sys_clk),
    .rst_n(key_n),
    .i2c_exec(i2c_exec),
    .bit_ctrl(1'b1),
    .i2c_rh_wl(i2c_rh_wl),
    .i2c_addr(i2c_data[23:8]),
    .i2c_data_w(i2c_data[7:0]),
    .i2c_data_r(i2c_data_r),
    .i2c_done(i2c_done),
    .i2c_ack(),
    .scl(cam_scl),
    .sda(cam_sda),
    .dri_clk(i2c_dri_clk)
);

endmodule
