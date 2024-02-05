module sc_apc_neuron
  #(parameter K = 3,
    parameter N = 2**K,
    parameter S = K + 2
    // increase S to 2*N make all output become zero
   )
  ( input logic clk,
    input logic reset,
    // input logic bias,
    input logic [N-1:0] din,
    input logic [N-1:0] weight, 
    output logic [N-1:0] mem1,
    output logic [K-1:0] count,
    output logic [S-1:0] current_state,
    output logic dout              // Output signal
  );

// reg [N -1:0] mem1 = 0;
// assign din & weight;
always_comb begin
  mem1 = ~(din  ^ weight) ;

end
accumulator #(.K(K)) accumulator_inst (
        .clk(clk),
        .data_in(mem1),
        .count(count)
      );

// always @(posedge clk) begin
//   count = 0;
//   for (int i = 0; i < N; i = i + 1) begin
//           count = count + mem1[i];
//   end
// end


// reg [S-1:0] current_state, next_state;
reg [S-1:0] next_state;

always @(negedge clk, negedge reset) begin
    if (reset)
        current_state <= (2**S)/2; // Initial state
    else
        current_state <= next_state;
        dout = current_state[S-1] ;
        
end

always @(current_state, count) begin
    if ((count > N/2) & (2**S-1 - current_state < count - N/2))next_state =  2**S-1;
    else if ((count < N/2) & (current_state < - count + N/2))
      next_state =  0;
    else next_state = current_state + count - N/2;

    // dout = (current_state[S-1] == 1'b1); 
end


endmodule

