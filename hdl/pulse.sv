module sc_pulse(
    input logic clk,           // Clock signal
    input logic rst,           // Reset signal
    input logic pulse_input,   // Input pulse signal
    output logic bit_stream    // Output bit stream
);
    logic [15:0] count = 16'b0;  // Counter to control the output bit duration
    logic pulse_edge;           // Detect positive edge of pulse_input

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            bit_stream <= 1'b0;  // Reset bit_stream
            count <= 16'b0;      // Reset the counter
            pulse_edge <= 1'b0;  // Reset the positive edge detector
        end else begin
            // Detect positive edge of pulse_input
            if (pulse_input && !pulse_edge) begin
                pulse_edge <= 1'b1;
            end else begin
                pulse_edge <= 1'b0;
            end

            if (pulse_edge) begin
                bit_stream <= 1'b1;  // Set bit_stream to 1 on positive edge
                count <= 0;           // Reset the counter
            end else if (count < 100) begin
                bit_stream <= 1'b1; 
                count <= count + 1;
            end else begin
                bit_stream <= 1'b0;  // Set bit_stream to 0
            end
        end
    end
endmodule
