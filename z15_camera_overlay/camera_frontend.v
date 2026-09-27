module camera_frontend(
 input clk_50m,input resetn,input cam_pclk,input cam_vsync,input cam_href,input [7:0] cam_data,
 output cam_rst_n,output cam_pwdn,output cam_scl,inout cam_sda,
 output led_init,output led_pclk,
 output reg [15:0] m_axis_tdata,output reg m_axis_tvalid,input m_axis_tready,
 output reg m_axis_tuser,output reg m_axis_tlast
);
wire i2c_clk,i2c_done,i2c_exec,i2c_rw,init_done; wire [7:0] i2c_rd; wire [23:0] i2c_data;
reg [25:0] pcnt; reg byte_sel,sof_pending; reg [7:0] hi; reg [9:0] xpos;
assign cam_pwdn=1'b0; assign cam_rst_n=1'b1; assign led_init=init_done; assign led_pclk=pcnt[25];
always @(posedge cam_pclk or negedge resetn) begin
 if(!resetn) begin pcnt<=0; byte_sel<=0; sof_pending<=1; xpos<=0; m_axis_tvalid<=0; m_axis_tuser<=0; m_axis_tlast<=0; end
 else begin
  pcnt<=pcnt+1'b1;
  if(m_axis_tvalid && m_axis_tready) m_axis_tvalid<=0;
  if(cam_vsync) begin sof_pending<=1; xpos<=0; byte_sel<=0; end
  else if(!cam_href) begin xpos<=0; byte_sel<=0; end
  else if(!byte_sel) begin hi<=cam_data; byte_sel<=1; end
  else begin
   byte_sel<=0;
   if(!m_axis_tvalid || m_axis_tready) begin
    m_axis_tdata<={hi,cam_data}; m_axis_tvalid<=1; m_axis_tuser<=sof_pending; m_axis_tlast<=(xpos==639);
    sof_pending<=0; xpos <= (xpos==639) ? 0 : xpos+1'b1;
   end
  end
 end
end
i2c_ov5640_rgb565_cfg cfg(.clk(i2c_clk),.rst_n(resetn),.i2c_data_r(i2c_rd),.i2c_done(i2c_done),
 .cmos_h_pixel(13'd640),.cmos_v_pixel(13'd480),.total_h_pixel(13'd1856),.total_v_pixel(13'd984),
 .i2c_exec(i2c_exec),.i2c_data(i2c_data),.i2c_rh_wl(i2c_rw),.init_done(init_done));
i2c_dri #(.SLAVE_ADDR(7'h3c),.CLK_FREQ(27'd50_000_000),.I2C_FREQ(20'd250_000)) i2c(
 .clk(clk_50m),.rst_n(resetn),.i2c_exec(i2c_exec),.bit_ctrl(1'b1),.i2c_rh_wl(i2c_rw),
 .i2c_addr(i2c_data[23:8]),.i2c_data_w(i2c_data[7:0]),.i2c_data_r(i2c_rd),.i2c_done(i2c_done),.i2c_ack(),
 .scl(cam_scl),.sda(cam_sda),.dri_clk(i2c_clk));
endmodule
