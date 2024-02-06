`timescale 1us/1ns

module parallel_lfsr_8bit_8x
#( 
    parameter NUM_OUTPUTS = 128,
    parameter SEED = 8'b0000_0001
)(
    input wire clk,
    input wire reset,
    output reg [7:0] lfsr = SEED,
    output wire [7:0] random[0:NUM_OUTPUTS-1]
);

// Intermediate wires for feedback calculations and intermediate states
wire [7:0] intermediate[0:NUM_OUTPUTS-2]; // Intermediate states, one less than NUM_OUTPUTS
wire feedback[0:NUM_OUTPUTS-1]; // Feedback values, equal to NUM_OUTPUTS

// Initial feedback calculation
assign feedback[0] = lfsr[7] ^ lfsr[6];

// Generate block to create intermediate states and calculate feedback
generate
    genvar i;
    for (i = 0; i < NUM_OUTPUTS-1; i = i + 1) begin : generate_intermediates
        if (i == 0) begin
            // First intermediate state based on initial LFSR state
            assign intermediate[i] = {lfsr[6:0], feedback[i]};
        end else begin
            // Subsequent intermediate states based on previous intermediates
            assign intermediate[i] = {intermediate[i-1][6:0], feedback[i]};
        end
        // Calculate feedback for next stage, if not the last iteration
        assign feedback[i+1] = intermediate[i][7] ^ intermediate[i][6];

    end
endgenerate

// Assign outputs from intermediate states and final LFSR state
generate
    for (i = 0; i < NUM_OUTPUTS-1; i = i + 1) begin : assign_outputs
        assign random[i] = intermediate[i];
    end
    assign random[NUM_OUTPUTS-1] = {intermediate[NUM_OUTPUTS-2][6:0], feedback[NUM_OUTPUTS-1]};
endgenerate

always @(posedge clk or posedge reset) begin
    if (reset) begin
        lfsr <= 8'b0000_0001; // Reset LFSR to initial state
    end else begin
        // Update LFSR with the last feedback
        lfsr <= {intermediate[NUM_OUTPUTS-2][6:0], feedback[NUM_OUTPUTS-1]};
    end
end

endmodule

