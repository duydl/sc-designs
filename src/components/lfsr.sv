module lfsr (
    input wire clk,
    input wire rst,
    output wire [7:0] lfsr_out
);

    reg [7:0] lfsr_reg;

    always @(posedge clk or posedge rst)
    begin
        if (rst)
            lfsr_reg <= 8'hFF;  // Initialize LFSR to a non-zero value on reset
        else if (clk)
            lfsr_reg <= {lfsr_reg[6:0], lfsr_reg[7] ^ lfsr_reg[5]};  // XOR feedback taps: 7 and 5
    end

    assign lfsr_out = lfsr_reg;

endmodule
