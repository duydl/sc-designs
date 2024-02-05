// sc_tanh.sv

`timescale 1us/1ns


// module sc_abs(
//     input logic clk,
//     input logic reset,
//     input logic x,
//     output logic y
// );

// typedef enum logic [3:0] {
//     S0 = 4'b0000,
//     S1 = 4'b0001,
//     S2 = 4'b0010,
//     S3 = 4'b0011,
//     S4 = 4'b0100,
//     S5 = 4'b0101,
//     S6 = 4'b0110,
//     S7 = 4'b0111,
//     S8 = 4'b1000,
//     S9 = 4'b1001,
//     S10 = 4'b1010,
//     S11 = 4'b1011,
//     S12 = 4'b1100,
//     S13 = 4'b1101,
//     S14 = 4'b1110,
//     S15 = 4'b1111
// } fsm_state;

// fsm_state current_state, next_state;
// wire is_smaller_8;

// always @(posedge clk, posedge reset) begin
//     if (reset)
//         current_state <= S7; // Initial state
//     else
//         current_state <= next_state;      
// end

// always @(current_state, x) begin
//     case (current_state)
//         S0: if (x) next_state = S1;
//         else next_state = S0;
//         S1: if (x) next_state = S2;
//         else next_state = S0;
//         S2: if (x) next_state = S3;
//         else next_state = S1;
//         S3: if (x) next_state = S4;
//         else next_state = S2;
//         S4: if (x) next_state = S5;
//         else next_state = S3;
//         S5: if (x) next_state = S6;
//         else next_state = S4;
//         S6: if (x) next_state = S7;
//         else next_state = S5;
//         S7: if (x) next_state = S8;
//         else next_state = S6;
//         S8: if (x) next_state = S9;
//         else next_state = S7;
//         S9: if (x) next_state = S10;
//         else next_state = S8;
//         S10: if (x) next_state = S11;
//         else next_state = S9;
//         S11: if (x) next_state = S12;
//         else next_state = S10;
//         S12: if (x) next_state = S13;
//         else next_state = S11;
//         S13: if (x) next_state = S14;
//         else next_state = S12;
//         S14: if (x) next_state = S15;
//         else next_state = S13;
//         S15: if (x) next_state = S15;
//         else next_state = S14;
//     endcase

//     if (current_state[3] == 1'b0)
//         y = (current_state[0] == 1'b0);
//     else
//         y = (current_state[0] == 1'b1);
// end

    
// endmodule



module sc_abs (
    input logic clk,
    input logic reset,
    input logic x,
    output logic y
);

parameter S = 6;
reg [S-1:0] current_state, next_state;

always @(posedge clk, posedge reset) begin
    if (reset)
        current_state <= (2**S)/2; // Initial state
    else
        current_state <= next_state;
end

always @(current_state, x) begin
    case (current_state)
        0: if (x) next_state = 1; 
            else next_state = 0;

        (2**S-1): if (x) next_state = (2**S-1);         
                        else next_state = (2**S-1) - 1;

        default: if (x) next_state = current_state + 1;
                 else next_state = current_state - 1;
    endcase
    if (current_state[S-1] == 1'b0)
        y = (current_state[0] == 1'b0);
    else
        y = (current_state[0] == 1'b1);
end

endmodule