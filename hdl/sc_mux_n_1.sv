module sc_mux_n_1
  #(parameter K = 3
   )
  ( 
    input wire [2**K -1:0] din, // Input data signals
    input wire [2**K -1:0] weight, 
    input wire [K-1:0] sel,      // Select signal
    output wire dout              // Output signal
  );
  reg [2**K -1:0] mem1;
  // can not select part of scalar: mem1 if reg do not defined width
  reg mem2;
  always_comb
  begin
    mem1 = weight & din;
    mem2 = mem1[sel];
  end

  assign dout = mem2;
endmodule