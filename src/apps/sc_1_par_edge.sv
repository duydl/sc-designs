`timescale 1us/1ns

module sc_par_edge #(
  parameter m = 32,
  parameter n = 32
  )(
    input logic clk,
    input logic reset,
    input logic sel,
    input logic  pixels [0:m*n-1], // m by n input pixels
    output logic  s [0:m*n-1]// m by n output signals
);

    genvar i, j;
    generate
        for (i = 0; i < m-1; i = i + 1) begin
            for (j = 0; j < n-1; j = j + 1) begin
                sc_robert_op_xor robert_op (
                    .clk(clk),
                    .rst(reset),
                    .r00(pixels[(i)*(n)+j]),
                    .r01(pixels[(i)*(n)+j+1]),
                    .r10(pixels[(i+1)*(n)+j]),
                    .r11(pixels[(i+1)*(n)+j+1]), 
                    .sel(sel), 
                    .s(s[(i)*(n)+j]) 
                );
            end
        end
    endgenerate

endmodule