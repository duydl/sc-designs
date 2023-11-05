module sc_pulse(
    input logic clk,           // Clock signal
    input logic rst,           // Reset signal
    input logic pulse_input,   // Input pulse signal
    output logic bit_stream    // Output bit stream
);
    logic [15:0] count = 16'b0;  // Counter to control the output bit duration

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            bit_stream <= 1'b0;  // Reset bit_stream
            count <= 16'b0;      // Reset the counter
        end else begin
            if (pulse_input && count == 100) begin
                bit_stream <= 1'b1;  // Set bit_stream to 1
                count <= 0;
                // if (count == 100) begin
                //     count <= 0;  // Increment the counter
                // end
            end else if (count < 100) begin
                bit_stream <= 1'b1; 
                count <= count + 1;
            end else begin
                bit_stream <= 1'b0;  // Set bit_stream to 0
            end
        end
    end
endmodule