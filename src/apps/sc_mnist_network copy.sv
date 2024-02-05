module sc_mnist_network_old
  #(
    // Input
    parameter N0 = 784,
    
    // First layer
    parameter K1 = 10, // K1 should be chosen such that 2^K1 >= N0 (input size)
    parameter N1 = 128,

    // Second layer - Output
    parameter K2 = 7,  // K2 should be chosen such that 2^K2 >= N1 (output of the first layer)
    parameter N2 = 10
    
  )
  (
    input wire clk,
    input wire reset,

    input wire [N0-1:0] din, // Input size

    input wire [N0-1:0] weight_0 [0:N1-1],
    input wire [K1:0] sel1 [0:N1-1], //select signal for layer 1

    input wire [N1-1:0] weight_1 [0:N2-1], 
    input wire [K2:0] sel2 [0:N2-1], //select signal for layer 2

    output wire [N2-1:0] dout // Output size
  );

  wire [N1-1:0] layer1_output; // Output of the first layer
  // Instantiate N1 neurons for the first layer
  generate
    genvar i;
    for (i = 0; i < N1; i = i + 1) begin
      sc_mux_neuron #(.K(K1)) neuron1_inst (
        .clk(clk),
        .reset(reset),
        .din(din),
        .weight(weight_0[i]),
        .sel(sel1[i]),
        .dout(layer1_output[i])
      );
    end
  endgenerate

  wire [N2-1:0] layer2_output; // Output of the second layer
  // Instantiate N2 neurons for the second layer
  generate
    genvar j;
    for (j = 0; j < N2; j = j + 1) begin
      sc_mux_n_1 #(.K(K2)) neuron2_inst (
        .din(layer1_output),
        .weight(weight_1[j]),
        .sel(sel2[j]),
        .dout(layer2_output[j])
      );
    end
  endgenerate

  // reg mem;
  // always @(posedge clk, posedge reset) begin
  //   if (reset)
  //     mem <= 0; // Initial state
  //   else
  //     mem <= layer2_output;      
  //   end

  assign dout = layer2_output;
endmodule