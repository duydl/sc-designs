module accumulator (
    input wire clk,
    input logic [7:0] data_in, // Adjust the width based on your parameter
    output logic [3:0] count
);

parameter NUM_BITS = 8; // Change this parameter to match the number of input bits

reg [3:0] count_tem = 0;

 // Initialize the count to zero

generate
    genvar i;
    for (i = 0; i < NUM_BITS; i = i + 1) begin : counter_loop
        always @(posedge data_in[i]) begin
            if (data_in[i]) begin
                count_tem = count_tem + 1;
            end
        end
    end
endgenerate

assign count = count_tem;

endmodule