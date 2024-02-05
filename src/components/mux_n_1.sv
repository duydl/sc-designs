module mux_2_1 
 #( parameter K = 3,
    parameter N = 2**K   // Data width of each input
  )
  ( input logic clk,
    input logic [N-1:0] din, // Input data signals
    input logic [K-1:0] sel,      // Select signal
    output logic dout              // Output signal
  );

    assign dout = din[sel];

endmodule