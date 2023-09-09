module sc_mnist_network
  #(
    // Input
    // parameter N1 = 784,
    parameter N0 = 100,
    
    // First layer
    parameter K1 = 7, // K1 should be chosen such that 2^K1 >= N0 (input size)
    parameter N1 = 16,

    // Output
    parameter K2 = 4,  // K2 should be chosen such that 2^K2 >= N1 (output of the first layer)
    parameter N2 = 4
    
  )
  (
    input wire clk,
    input wire reset,

    input wire sel1, //select signal for layer 1
    input wire sel2, //select signal for output

    input wire [N0-1:0] din, // Input size

    input wire [N1-1:0] weight_0 [0:N0-1],

    input wire [N2-1:0] weight_1 [N1-1:0], 

    output wire [N2-1:0] dout // Output size
  );

  wire [N1-1:0] layer1_output; // Output of the first layer
  // Instantiate 128 neurons for the first layer
  generate
    genvar i;
    for (i = 0; i < N1; i = i + 1) begin
      sc_mux_neuron #(.K(K1)) neuron1_inst (
        .clk(clk),
        .reset(reset),
        .din(din),
        .weight(weight_0[i]),
        .sel(sel),
        .dout(layer1_output[i])
      );
    end
  endgenerate

  wire [N2-1:0] layer2_output; // Output of the second layer
  // Instantiate 10 neurons for the second layer
  generate
    genvar j;
    for (j = 0; j < N2; j = j + 1) begin
      sc_mux_n_1 #(.K(K2)) neuron2_inst (
        .din(layer1_output),
        .weight(weight_1[j]),
        .sel(sel),
        .dout(layer2_output[j])
      );
    end
  endgenerate

  assign dout = layer2_output;

endmodule