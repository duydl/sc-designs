module lfsr #(
    parameter N = 16,    // Number of bits in LFSR
    parameter SEED = 1   // Initial seed value
) (
    input wire clk,
    input wire reset,
    output wire [N-1:0] lfsr_out
);

    reg [N-1:0] lfsr_state;
    wire feedback;

    // Feedback logic (XOR of selected bits)
    assign feedback = lfsr_state[N-1] ^ lfsr_state[N-2] ^ lfsr_state[0];

    always @(posedge clk or posedge reset) begin
        if (reset)
            lfsr_state <= SEED; // Initial seed
        else
            lfsr_state <= {lfsr_state[N-2:0], feedback}; // Shift and feedback
    end

    assign lfsr_out = lfsr_state;

endmodule