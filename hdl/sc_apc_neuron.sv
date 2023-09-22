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
reg [K:0] count_tem = 0;
// reg [K-1:0] count = 0; why error
reg [K:0] count ;


always @(din, weight) begin
  mem1 = din & weight;
end

accumulator #(.K(K)) accumulator_inst (
        .clk(clk),
        .data_in(mem1),
        .count(count)
      );
// generate
//   genvar i;
//   for (i = 0; i < N; i = i + 1) begin : counter_loop
//       always @(posedge mem1[i]) begin
//           if (mem1[i]) begin
//               count_tem = count_tem + 1;
//           end
//       end
//   end
// endgenerate

// assign count = count_tem;

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

    dout = (current_state[S-1] == 1'b1); 
    
end



endmodule





// count - N/2 could overflow
// NOT OK
// always @(current_state, count) begin
//   if (2**S-1 - current_state + count - N/2 > - count + N/2)
//     next_state =  2**S-1;
//   else if (current_state < - count + N/2)
//     next_state =  0;
//   else next_state = current_state + count - N/2;

//   dout = (current_state[S-1] == 1'b1); 
  
// end



// INFO     cocotb:simulator.py:302 current state 11110100
// INFO     cocotb:simulator.py:302 count 1000
// INFO     cocotb:simulator.py:302 N/2 8
// INFO     cocotb:simulator.py:302 current state 11111000
// INFO     cocotb:simulator.py:302 count 1000
// INFO     cocotb:simulator.py:302 N/2 8
// INFO     cocotb:simulator.py:302 current state 11111100
// INFO     cocotb:simulator.py:302 count 1000
// INFO     cocotb:simulator.py:302 N/2 8
// INFO     cocotb:simulator.py:302 current state 00000000
// INFO     cocotb:simulator.py:302 count 1000
// INFO     cocotb:simulator.py:302 N/2 8
// INFO     cocotb:simulator.py:302 current state 00000100