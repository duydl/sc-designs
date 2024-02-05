module q_value_updater (
    input clk,
    input reset,
    input q_bitstream,        // Bitstream for Q-value
    input q_prime_bitstream,  // Bitstream for Q'-value (next state)
    input reward_bitstream,   // Bitstream for reward
    input alpha_bitstream,    // Bitstream for alpha
    input gamma_bitstream,    // Bitstream for gamma
    output [1:0] q_updated_bitstream // Updated Q-value bitstream
);

wire product1, product2, product3;

// Multiplications using XOR gates
assign product1 = (~alpha) ^ q_bitstream;      // (1-alpha) * q
assign product2 = alpha_bitstream ^ reward_bitstream; // alpha * reward
assign product3 = gamma_bitstream ^ alpha_bitstream & q_prime_bitstream; // gamma * alpha * q'

assign q_updated_bitstream = product1 + product2 + product3
endmodule



module SDconverter_2in (
    input clk,
    input reset,
    input bitstream1,              // First serial bitstream input
    input bitstream2,              // Second serial bitstream input
    output reg [N-1:0] binary_out, // Combined binary output
    output reg valid               // Flag to indicate valid output
);

parameter N = 8;                   // Width of the binary output
parameter BITSTREAM_LENGTH = 256;  // Length of each bitstream for counting

reg [N-1:0] count;
reg [$clog2(BITSTREAM_LENGTH)-1:0] bit_count; // Counter for the number of bits processed

always @(posedge clk or posedge reset) begin
    if (reset) begin
        count <= 0;
        bit_count <= 0;
        valid <= 0;
    end else begin
        if (bit_count < BITSTREAM_LENGTH) begin
            // Count the number of '1's in both bitstreams
            count <= count + bitstream1 + bitstream2;
            bit_count <= bit_count + 1;
            valid <= 0;
        end else begin
            // Output the combined count as binary after BITSTREAM_LENGTH bits
            binary_out <= count;
            count <= 0;
            bit_count <= 0;
            valid <= 1; // Indicate that the output is valid
        end
    end
end

endmodule