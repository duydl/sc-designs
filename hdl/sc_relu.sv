module sc_relu (
    input clk,    // Clock
    input reset,  // Asynchronous reset active low
    input x,
    output y    
);
    parameter DEPTH = 4;

    logic overHalf;
    logic [DEPTH:0] cnt;
    logic inc;
    logic dec;

    assign inc = x;
    assign dec = ~x;

    always_ff @(posedge clk or negedge reset) begin : proc_cnt
        if(~reset) begin
            cnt <= {{1'b1}, {(DEPTH-1){1'b0}}};
        end else begin
            if(inc & ~dec & ~&cnt) begin
                cnt <= cnt + 1;
            end else if(~inc & dec & ~|cnt) begin
                cnt <= cnt - 1;
            end else begin
                cnt <= cnt;
            end
        end
    end

    assign overHalf = cnt[DEPTH-1];

    assign y = overHalf ? x : 0;

endmodule