`timescale 1us/1ns

module sc_par_noise #(
  parameter m = 32,
  parameter n = 32
  )(
    input logic clk,
    input logic reset,
    input logic  in_bits [0:m*n-1], // m by n input pixels
    output logic out_bits [0:m*n-1]// m by n output signals
);

genvar i, j;
generate
    for (i = 0; i < m-2; i = i + 1) begin
        for (j = 0; j < n-2; j = j + 1) begin
        wire window [0:8];
        assign window[0] = in_bits[i*n + j];
        assign window[1] = in_bits[i*n + j+1];
        assign window[2] = in_bits[i*n + j+2];
        assign window[3] = in_bits[(i+1)*n + j];
        assign window[4] = in_bits[(i+1)*n + j+1];
        assign window[5] = in_bits[(i+1)*n + j+2];
        assign window[6] = in_bits[(i+2)*n + j];
        assign window[7] = in_bits[(i+2)*n + j+1];
        assign window[8] = in_bits[(i+2)*n + j+2];

        sc_median_3x3 median_fil (
            .window(window),
            .result(out_bits[(i+1)*(n)+j+1]) 
        );
        end
    end
endgenerate

endmodule