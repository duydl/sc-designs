module sc_apc_neuron
  #(parameter K = 3,
    parameter N = 2**K,
    parameter S = 8
   )
  ( input logic clk,
    input logic reset,
    input logic [N-1:0] din,
    input logic [N-1:0] weight, 
    output logic dout              // Output signal
  );

reg [N -1:0] mem1;
reg [K:0] count;

assign mem1 = din & weight;

// accumulator #(.K(K)) accumulator_inst (
//         .clk(clk),
//         .data_in(mem1),
//         .count(count)
//       );

always @(posedge clk) begin
  count = 0;
  for (int i = 0; i < N; i = i + 1) begin

          count = count + mem1[i];

  end
end

reg [S-1:0] current_state, next_state;

always @(posedge clk, posedge reset) begin
    if (reset)
        current_state <= (2**S)/2; // Initial state
    else
        current_state <= next_state;
end

always @(current_state, count) begin
    if ((count > N/2) & (2**S-1 - current_state < count - N/2))next_state =  2**S-1;
    else if ((count < N/2) & (current_state < - count + N/2))
      next_state =  0;
    else next_state = current_state + count - N/2;

    dout = current_state[S-1]; 
end


endmodule

