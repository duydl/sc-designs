// module accumulator
// #(
//     parameter K = 3,
//     parameter N = 2**K
// )
// (
//     input wire clk,
//     input wire reset,
//     input logic [N-1:0] data_in,
//     output logic [K:0] count = 0
// );

//     // Initialize the count to zero


//     always @(posedge clk or posedge reset) begin
//         if (reset) begin
//             count <= 0; // Reset the count to zero
//         end else begin
//             // Count logic
//             count <= 0;
//             for (int i = 0; i < N; i = i + 1) begin
//                     count <= count + data_in[i];
//             end
//         end
//     end



// endmodule


// module accumulator 
// #(  parameter K = 3,
//     parameter N = 2**K

// )
// (
//     input wire clk,
//     input logic [N-1:0] data_in, // Adjust the width based on your parameter
//     output logic [K:0] count = 0
// );


// generate
//     genvar i;
//     for (i = 0; i < N; i = i + 1) begin 
//         always @(posedge data_in[i]) begin
//                 count = count + data_in[i];
//         end
//     end
// endgenerate


// endmodule


module accumulator 
#(  parameter K = 3,
    parameter N = 2**K

)(
    input wire clk,
    input logic [N-1:0] data_in, // Adjust the width based on your parameter
    output logic [K:0] count 
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




