`timescale 1us/1ns

module mux_2_1 (
    input logic clk,
    input logic a,
    input logic b,
    input logic select,
    output logic c
  );

  assign  c = select ? a : b;

endmodule
