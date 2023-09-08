// sc_add.sv

`timescale 1us/1ns


module sc_sub (
    input logic clk,
    input logic a,
    input logic b,
    input logic select,
    output logic c
  );


  always @(posedge clk)
  begin
    c <= select ? a : ~b;
  end


endmodule
