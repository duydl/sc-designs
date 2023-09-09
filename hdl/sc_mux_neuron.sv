module sc_mux_neuron
  #(parameter K = 3
   )
  ( input wire clk,
    input wire reset,
    input wire [2**K-1:0] din,
    input wire [2**K-1:0] weight, 
    input wire [K-1:0] sel,      // Select signal
    output wire dout              // Output signal
  );

  reg [2**K -1:0] mem1;
  reg mem2;
//   warning: always_comb process has no sensitivities. if mem1 is wire. not reg.
//   mem2 is not a valid l-value in if mem2 is wire not reg
  // Search reg vs wire
  always_comb
  begin
    mem1 = din & weight;
    mem2 = mem1[sel];
  end

  sc_tanh tanh_x (
    .clk(clk),
    .reset(reset),
    .x(mem2),
    .y(dout)
);

endmodule