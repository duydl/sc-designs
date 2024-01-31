module edge_detector_deterministic (
    input logic clk,
    input logic rst,
    input logic r00, r01, r10, r11, 
    input logic sel,
    output logic s
);

    logic t0, t1;
    logic a0, a1;
    logic result;
    logic r00_pulse, r01_pulse, r10_pulse, r11_pulse;

    sc_pulse pulse_module_r00 (
        .clk(clk),
        .rst(rst),
        .pulse_input(r00),
        .bit_stream(r00_pulse)
    );

    sc_pulse pulse_module_r01 (
        .clk(clk),
        .rst(rst),
        .pulse_input(r01),
        .bit_stream(r01_pulse)
    );

    sc_pulse pulse_module_r10 (
        .clk(clk),
        .rst(rst),
        .pulse_input(r10),
        .bit_stream(r10_pulse)
    );

    sc_pulse pulse_module_r11 (
        .clk(clk),
        .rst(rst),
        .pulse_input(r11),
        .bit_stream(r11_pulse)
    );

    sc_sub sub_x0 (
        .clk(clk),
        .a(r00_pulse),
        .b(r11_pulse),
        .select(sel),
        .c(t0)
    );

    sc_sub sub_x1 (
        .clk(clk),
        .a(r10_pulse),
        .b(r01_pulse),
        .select(sel),
        .c(t1)
    );

    sc_abs abs_x (
        .clk(clk),
        .reset(rst),
        .x(t0),
        .y(a0)
    );

    sc_abs abs_y (
        .clk(clk),
        .reset(rst),
        .x(t1),
        .y(a1)
    );

    sc_add add_results (
        .clk(clk),
        .a(a0),
        .b(a1),
        .select(sel),
        .c(result)
    );

    assign s = result;

endmodule