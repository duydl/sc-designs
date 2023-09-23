module sc_mnist_network
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
    input logic clk,
    input logic reset,

    input logic [N0-1:0] din, // Input size

    input logic [N0-1:0] weight_0 [0:N1-1],


    input logic [N1-1:0] weight_1 [0:N2-1], 


    output logic [N2-1:0] dout // Output size
  );

  reg [N1-1:0] layer1_output; // Output of the first layer
  reg [N0-1:0] mem1 [0:N1-1];
  reg [7:0] states [0:N1-1];
  reg [K1:0] count [0:N1-1];
  // Instantiate N1 neurons for the first layer
  generate
    genvar i;
    for (i = 0; i < N1; i = i + 1) begin
      sc_apc_neuron #(.K(K1)) neuron1_inst (
        .clk(clk),
        .reset(reset),
        .din(din),
        .weight(weight_0[i]),
        .count(count[i]),
        .mem1(mem1[i]),
        .current_state(states[i]),
        .dout(layer1_output[i])
      );
    end
  endgenerate

  reg [N2-1:0] layer2_output; // Output of the second layer
  // Instantiate N2 neurons for the second layer
  generate
    genvar j;
    for (j = 0; j < N2; j = j + 1) begin
      sc_apc_neuron #(.K(K2)) neuron2_inst (
        .clk(clk),
        .reset(reset),
        .din(layer1_output),
        .weight(weight_1[j]),
        .dout(layer2_output[j])
      );
    end
  endgenerate

  assign dout = layer2_output;
  // always @(posedge clk, posedge reset) begin
  //   if (reset)
  //     dout <= 0; // Initial state
  //   else
  //     dout <= layer2_output;      
  //   end

endmodule