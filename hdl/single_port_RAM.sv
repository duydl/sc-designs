module single_port_RAM
#(
    parameter addr_width = 7,
                 data_width = 10
)
(
    input wire clk,
    input wire we,
    input wire [addr_width-1:0] addr,
    input wire [data_width-1:0] din,
    output wire [data_width-1:0] dout
);

reg [data_width-1:0] ram_single_port[2**addr_width-1:0];

always @(posedge clk)
begin
    if (we == 1) // write data to address 'addr'
        ram_single_port[addr] <= din;
end

// read data from address 'addr'
assign dout = ram_single_port[addr];

endmodule 