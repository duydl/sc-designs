// Nsmv.sv

`timescale 1us/1ns


module Nsmv(
    input wire clk,
    input wire reset,
    input wire in,
    output wire out,
);
    parameter N = 64;  // Number of states

    reg [7:0] state;

    always @(posedge clk or posedge reset) begin
        if (reset)
            state <= 0;
        else if (in) begin
            if (state == N-1)
                state <= state;
            else
                state <= state + 1;
        end else begin
            if (state == 0)
                state <= 0;
            else
                state <= state - 1;
        end
    end

    assign out = (state < N/2) ? 1'b1 : 1'b0;
endmodule