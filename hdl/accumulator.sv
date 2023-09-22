module accumulator 
#(  parameter K = 3,
    parameter N = 2**K

)
(
    input wire clk,
    input logic [N-1:0] data_in, // Adjust the width based on your parameter
    output logic [K:0] count
);
// Initialize the count to zero
reg [K:0] count_tem = 0;

generate
    genvar i;
    for (i = 0; i < N; i = i + 1) begin : counter_loop
        always @(posedge data_in[i]) begin
            if (data_in[i]) begin
                count_tem = count_tem + 1;
            end
        end
    end
endgenerate

assign count = count_tem;

endmodule