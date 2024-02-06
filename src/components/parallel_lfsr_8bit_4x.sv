`timescale 1us/1ns

module parallel_lfsr_8bit_4x
#( 
    parameter NUM_OUTPUTS = 128,
    parameter SEED = 8'b0000_0001
)(
    input wire clk,    // Clock input
    input wire reset,  // Asynchronous reset
    output reg [7:0] lfsr = SEED, // Initialize LFSR with a non-zero value
    output wire [7:0] random1,       // First random number
    output wire [7:0] random2,       // Second random number
    output wire [7:0] random3,       // Third random number
    output wire [7:0] random4        // Fourth random number
);

// Intermediate wires for feedback calculations and intermediate states
wire [7:0] intermediate[2:0]; // Array to hold intermediate states, adjusted for indexing
wire feedback[3:0]; // Array to hold feedback values

// Initial feedback calculation
assign feedback[0] = lfsr[7] ^ lfsr[6];

// Generate intermediate states and subsequent feedbacks
assign intermediate[0] = {lfsr[6:0], feedback[0]};
assign feedback[1] = intermediate[0][7] ^ intermediate[0][6];

assign intermediate[1] = {intermediate[0][6:0], feedback[1]};
assign feedback[2] = intermediate[1][7] ^ intermediate[1][6];

assign intermediate[2] = {intermediate[1][6:0], feedback[2]};
assign feedback[3] = intermediate[2][7] ^ intermediate[2][6];

// Assigning intermediate states to random outputs
assign random1 = intermediate[0];
assign random2 = intermediate[1];
assign random3 = intermediate[2];
assign random4 = {intermediate[2][6:0], feedback[3]}; // Final state as random number

always @(posedge clk or posedge reset) begin
    if (reset) begin
        lfsr <= 8'b0000_0001; // Reset LFSR
    end else begin
        // Update LFSR with the final feedback
        lfsr <= {intermediate[2][6:0], feedback[3]};
    end
end

endmodule