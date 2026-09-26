`timescale 1ns/1ps
//
// tb_cnn_top：对 cnn_top 做十张固定图片的自检。
//
// 测试向量由 module/scripts/生成仿真数据.py 从
// B_模型交接_v1/03_FPGA交接/hardware_reference.npz 导出（即 A 应比对的
// 标准答案本身），逐张写图片 -> 启动 -> 等待 done -> 比较十个 int32 分数
// 和预测数字，全部一致才算通过。
//
// 运行前先准备好两组 .mem：
//   1. B_模型交接_v1/generated_fpga/*.mem   （运行 03_FPGA交接/导出FPGA参数.py 生成）
//   2. module/sim/tb_*.mem                  （运行 module/scripts/生成仿真数据.py 生成）
//
module tb_cnn_top;

    localparam IMG_BYTES = 784;
    localparam N_CLASS   = 10;
    localparam N_IMAGES  = 10;

    reg clk;
    reg rst_n;

    reg        img_wr_en;
    reg [9:0]  img_wr_addr;
    reg [7:0]  img_wr_data;
    reg        start;

    wire        busy;
    wire        done;
    wire signed [319:0] scores_packed;
    wire [3:0]           digit;

    cnn_top dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .img_wr_en     (img_wr_en),
        .img_wr_addr   (img_wr_addr),
        .img_wr_data   (img_wr_data),
        .start         (start),
        .busy          (busy),
        .done          (done),
        .scores_packed (scores_packed),
        .digit         (digit)
    );

    always #5 clk = ~clk; // 100MHz

    // ---------------- 测试向量 ----------------
    reg [7:0]  images     [0:N_IMAGES*IMG_BYTES-1];
    reg signed [31:0] exp_scores [0:N_IMAGES*N_CLASS-1];
    reg [7:0]  exp_digit  [0:N_IMAGES-1];

    initial begin
        $readmemh("tb_images.mem",      images);
        $readmemh("tb_scores.mem",      exp_scores);
        $readmemh("tb_predictions.mem", exp_digit);
    end

    integer img_idx, byte_idx, cls_idx;
    integer fail_count;
    reg signed [31:0] got_score;
    reg signed [31:0] want_score;

    // 所有驱动 DUT 输入的赋值都在时钟沿之后延迟 1 个时间单位再改变，
    // 避免和 DUT 内部同步在同一个仿真时刻抢跑（经典的测试激励竞争问题，
    // 用 input_buffer 单独测试复现确认过：不加这个延迟会导致隔一个地址
    // 丢一次写入）。
    task load_image;
        input integer idx;
        begin
            img_wr_en = 1'b1;
            for (byte_idx = 0; byte_idx < IMG_BYTES; byte_idx = byte_idx + 1) begin
                img_wr_addr = byte_idx[9:0];
                img_wr_data = images[idx*IMG_BYTES + byte_idx];
                @(posedge clk);
                #1;
            end
            img_wr_en = 1'b0;
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        img_wr_en = 1'b0;
        img_wr_addr = 10'd0;
        img_wr_data = 8'd0;
        start = 1'b0;
        fail_count = 0;

        repeat (3) @(posedge clk);
        #1;
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        #1;

        for (img_idx = 0; img_idx < N_IMAGES; img_idx = img_idx + 1) begin
            load_image(img_idx);

            @(posedge clk);
            #1;
            start = 1'b1;
            @(posedge clk);
            #1;
            start = 1'b0;

            wait (done == 1'b1);
            @(posedge clk); // done 拉高的这拍再等一拍，确保信号稳定采样

            for (cls_idx = 0; cls_idx < N_CLASS; cls_idx = cls_idx + 1) begin
                got_score  = scores_packed[32*cls_idx +: 32];
                want_score = exp_scores[img_idx*N_CLASS + cls_idx];
                if (got_score !== want_score) begin
                    fail_count = fail_count + 1;
                    $display("[FAIL] image=%0d class=%0d got=%0d want=%0d",
                              img_idx, cls_idx, got_score, want_score);
                end
            end

            if (digit !== exp_digit[img_idx][3:0]) begin
                fail_count = fail_count + 1;
                $display("[FAIL] image=%0d digit got=%0d want=%0d",
                          img_idx, digit, exp_digit[img_idx]);
            end else begin
                $display("[PASS] image=%0d digit=%0d (scores all matched)", img_idx, digit);
            end

            repeat (2) @(posedge clk);
        end

        if (fail_count == 0) begin
            $display("==== ALL %0d IMAGES PASSED ====", N_IMAGES);
        end else begin
            $display("==== %0d MISMATCHES ====", fail_count);
        end
        $finish;
    end

endmodule
