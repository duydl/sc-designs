

module sc_robert_op_lsm (
    input logic clk,
    input logic reset,
    input logic r00, r01, r10, r11, 
    input logic sel,
    output logic out
);

    logic t0, t1;
    logic a0, a1;

    mux_2_1 sub_x0 (
        .clk(clk),
        .a(r00),
        .b(~r11),
        .select(sel),
        .c(t0)
    );
    mux_2_1 sub_x1 (
        .clk(clk),
        .a(r10),
        .b(~r01),
        .select(sel),
        .c(t1)
    );

    sc_abs abs_x (
        .clk(clk),
        .reset(reset),
        .x(t0),
        .y(a0)
    );

    sc_abs abs_y (
        .clk(clk),
        .reset(reset),
        .x(t1),
        .y(a1)
    );

    mux_2_1 add (
        .clk(clk),
        .a(a0),
        .b(a1),
        .select(sel),
        .c(out)
    );

endmodule
