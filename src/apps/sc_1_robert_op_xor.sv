`timescale 1us/1ns

module sc_robert_op_xor(
    input logic r00, r01, r10, r11, 
    input logic sel,
    output logic result
);

    logic a0, a1;

    assign a0 = r00 ^ r11;

    assign a1 = r01 ^ r10;

    mux_2_1 add (
        .clk(clk),
        .a(a0),
        .b(a1),
        .select(sel),
        .c(result)
    );

endmodule
