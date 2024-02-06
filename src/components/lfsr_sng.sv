`timescale 1us/1ns

module lfsr_rng (
    input wire clk,
    input wire rst,
    input wire [7:0] threshold,
    output reg out_bit
);

    reg [7:0] lfsr_reg;

    always @(posedge clk or posedge rst) begin
        if (rst)
            lfsr_reg <= 8'hFF;  // Initialize LFSR to a non-zero value on reset
        else if (clk)
            lfsr_reg <= {lfsr_reg[6:0], lfsr_reg[7] ^ lfsr_reg[5]};  // LFSR feedback taps: 7 and 5

        // Comparator to generate the output bit based on probability
        if (lfsr_reg > threshold) // If LFSR value is greater than threshold, set output bit to 1
            out_bit <= 1'b1;
        else                      // Otherwise, set output bit to 0
            out_bit <= 1'b0;
    end

endmodule
