`timescale 1us/1ns

module edge_detector_deterministic_xor_abs (
    input logic clk,
    input logic rst,
    input logic r00, r01, r10, r11, 
    input logic sel,
    output logic s
);

    logic a0, a1;
    logic result;

    assign a0 = r00 ^ r11;

    assign a1 = r01 ^ r10;

    sc_add add_results (
        .clk(clk),
        .a(a0),
        .b(a1),
        .select(sel),
        .c(result)
    );

    assign s = result;

endmodule
