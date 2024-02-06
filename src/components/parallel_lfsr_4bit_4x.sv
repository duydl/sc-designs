`timescale 1us/1ns

module parallel_lfsr_4bit_4x
#( 
    parameter NUM_OUTPUTS = 128,
    parameter SEED = 4'b0001
)(
    input wire clk,    // Clock input
    input wire reset,  // Asynchronous reset
    output reg [3:0] lfsr = SEED, // Initialize LFSR with a non-zero value
    output wire [3:0] random1,       // First random number
    output wire [3:0] random2,       // Second random number
    output wire [3:0] random3,       // Third random number
    output wire [3:0] random4        // Fourth random number
);

// Intermediate wires for feedback calculations and intermediate states
wire [3:0] intermediate[2:0]; // Array to hold intermediate states, adjusted for indexing
wire feedback[3:0]; // Array to hold feedback values

// Initial feedback calculation
assign feedback[0] = lfsr[3] ^ lfsr[2];

// Generate intermediate states and subsequent feedbacks
assign intermediate[0] = {lfsr[2:0], feedback[0]};
assign feedback[1] = intermediate[0][3] ^ intermediate[0][2];

assign intermediate[1] = {intermediate[0][2:0], feedback[1]};
assign feedback[2] = intermediate[1][3] ^ intermediate[1][2];

assign intermediate[2] = {intermediate[1][2:0], feedback[2]};
assign feedback[3] = intermediate[2][3] ^ intermediate[2][2];

// Assigning intermediate states to random outputs
assign random1 = intermediate[0];
assign random2 = intermediate[1];
assign random3 = intermediate[2];
assign random4 = {intermediate[2][2:0], feedback[3]}; // Final state as random number

always @(posedge clk or posedge reset) begin
    if (reset) begin
        lfsr <= 4'b0001; // Reset LFSR
    end else begin
        // Update LFSR with the final feedback
        lfsr <= {intermediate[2][2:0], feedback[3]};
    end
end

endmodule
