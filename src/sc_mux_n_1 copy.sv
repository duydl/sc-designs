module sc_n_1_archive
  #(parameter K = 3,
    N = 2**K   // Data width of each input
   )
  ( input wire clk,
    input wire [N-1:0] din, // Input data signals
    input wire [K-1:0] sel,      // Select signal
    output wire dout              // Output signal
  );

  reg mem;
  always @(posedge clk)
  begin
    mem <= din[sel];
  end

  assign dout = mem;
endmodule