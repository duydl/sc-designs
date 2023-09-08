module d2p_converter #(
    parameter N = 16,       // Number of bits in LFSR
    parameter SEED = 1,     // Initial seed value
    parameter THRESHOLD = 0.5 // Probability threshold (0 to 1)
) (
    input wire clk,
    input wire reset,
    output wire probability
);

    wire [N-1:0] lfsr_out;
    wire [N-1:0] threshold_value;
    wire is_above_threshold;

    lfsr #(N, SEED) lfsr_inst (
        .clk(clk),
        .reset(reset),
        .lfsr_out(lfsr_out)
    );

    assign threshold_value = {N{THRESHOLD}};
    assign is_above_threshold = (lfsr_out >= threshold_value);

    assign probability = is_above_threshold;

endmodule