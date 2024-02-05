// Parallel Counter

module accumulator 
#(  parameter K = 3,
    parameter N = 2**K

)(
    input wire clk,
    input logic [N:0] data_in, // Adjust the width based on your parameter
    output logic [K+1:0] count 
);

    reg [K:0] count_tem = 0;

    always @(posedge clk) begin
        count_tem = 0;
        for (int i = 0; i < N; i = i + 1) begin
            count_tem = count_tem + data_in[i];
        end
    end

    assign count = count_tem;

endmodule




