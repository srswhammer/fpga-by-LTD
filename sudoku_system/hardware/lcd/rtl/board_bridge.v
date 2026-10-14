module board_bridge #(parameter TIMEOUT=5000000)(
 input clk,input pclk,input rst_n,input rx_done,input [7:0] rx_data,
 input frame_blank,output reg tx_en,output reg [7:0] tx_data,
 output reg [647:0] board);
`include "board_init.vh"
reg [647:0] staging;
reg [2:0] state;
reg [6:0] count;
reg [7:0] checksum;
reg invalid;
reg [23:0] timer;
reg request_toggle, applied_toggle;
(* ASYNC_REG="TRUE" *) reg req_sync1,req_sync2,ack_sync1,ack_sync2;
initial begin board=INITIAL_BOARD;staging=INITIAL_BOARD;end
always @(posedge pclk or negedge rst_n) begin
 if(!rst_n) begin req_sync1<=0;req_sync2<=0;applied_toggle<=0;board<=INITIAL_BOARD;end
 else begin
  req_sync1<=request_toggle;req_sync2<=req_sync1;
  if(frame_blank && req_sync2!=applied_toggle) begin
   board<=staging;applied_toggle<=req_sync2;
  end
 end
end
always @(posedge clk or negedge rst_n) begin
 if(!rst_n) begin
  staging<=INITIAL_BOARD;state<=0;count<=0;checksum<=0;invalid<=0;
  timer<=0;request_toggle<=0;ack_sync1<=0;ack_sync2<=0;tx_en<=0;tx_data<=0;
 end else begin
  ack_sync1<=applied_toggle;ack_sync2<=ack_sync1;tx_en<=0;
  if(state==4) begin
   if(ack_sync2==request_toggle) begin state<=0;tx_data<=8'h06;tx_en<=1;end
  end else if(rx_done) begin
   timer<=0;
   case(state)
    0: if(rx_data==8'h55) state<=1;
    1: if(rx_data==8'haa) begin state<=2;count<=0;checksum<=0;invalid<=0;end
       else if(rx_data!=8'h55) state<=0;
    2: begin
     staging[count*8+:8]<=rx_data;checksum<=checksum+rx_data;
     if(rx_data[7:5]!=0 || rx_data[3:0]>9) invalid<=1;
     if(count==80) state<=3;else count<=count+1'b1;
    end
    3: begin
     if(!invalid && rx_data==checksum) begin request_toggle<=~request_toggle;state<=4;end
     else begin state<=0;tx_data<=8'h15;tx_en<=1;end
    end
    default: state<=0;
   endcase
  end else if(state!=0) begin
   if(timer==TIMEOUT-1) begin state<=0;timer<=0;tx_data<=8'h15;tx_en<=1;end
   else timer<=timer+1'b1;
  end else timer<=0;
 end
end
endmodule
