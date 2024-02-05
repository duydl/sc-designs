module sc_apc_neuron_2
  #(parameter K = 3,
    parameter N = 2**K,
    parameter S = 6
   )
  ( input logic clk,
    input logic reset,
    input logic [N-1:0] din,
    input logic [N-1:0] weight, 
    output logic dout              // Output signal
  );

reg [N -1:0] mem1;
//   warning: always_comb process has no sensitivities. if mem1 is wire. not reg.
//   mem2 is not a valid l-value in if mem2 is wire not reg
  // Search reg vs wire

assign mem1 = din & weight;

reg [K-1:0] count_tem = 0;
// reg [K-1:0] count = 0; why error
reg [K-1:0] count ;

generate
    genvar i;
    for (i = 0; i < 2**K -1; i = i + 1) begin : counter_loop
        always @(posedge mem1[i]) begin
            if (mem1[i]) begin
                count_tem = count_tem + 1;
            end
        end
    end
endgenerate

assign count = count_tem;

reg [S-1:0] current_state, next_state;

always @(posedge clk, posedge reset) begin
    if (reset)
        current_state <= (2**S)/2; // Initial state
    else
        current_state <= next_state;
end
 

always @(current_state, count) begin
    case (current_state)
        0: if (count <= N/2) next_state = 0; 

        2**S-1: if (count >= N/2) next_state = (2**S-1) - 1;         

        default: next_state = current_state + count - N/2;
                 
    endcase
    dout = (current_state[S-1] == 1'b1); // Output based on odd/even state
end



endmodule