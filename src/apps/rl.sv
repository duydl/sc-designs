module SignalGenerator(
  input wire [3:0] x,  // Assuming x is a 4-bit value (0 to 15)
  output wire signal
);

  reg [3:0] on_time;

  // Update on_time based on the input value x
  always @* begin
    on_time = (x <= 4'd16) ? x : 4'd16;
  end

  // Generate signal based on duty cycle
  assign signal = (on_time > 0);

endmodule

module UnaryComputing(
  input wire [3:0] q_values [0:3], // Q-values for 4 actions
  output wire max_value // Output representing the max Q-value
);

  reg [3:0] unary_streams [0:3];
  reg [3:0] max_unary_stream;

  // Generate unary bitstreams for each action
  always @* begin
    for (int i = 0; i < 4; i = i + 1) begin
      unary_streams[i] = (q_values[i] > 0) ? {4{1'b1}} : 4'b0;
    end
  end

  // Pass unary bitstreams through OR gate to find the max
  always @* begin
    max_unary_stream = unary_streams[0] | unary_streams[1] | unary_streams[2] | unary_streams[3];
  end

  // Output the max Q-value based on the unary streams
  assign max_value = (max_unary_stream > 0);




  qtable qt0(
        .i_clk(clk),
        .i_rst(rst),
        .i_addr_r(addrr_q), 
        .i_addr_w(addrw_q),
        .i_read_en(rflag_q), 
        .i_write_en(wflag_q),
        .i_data(data_in_q),
        .o_data(data_out_qdata_out_q)); 

    policytable qmaxt0(
        .i_clk(clk),
        .i_rst(rst),
        .i_addr_r(addrr_qmax), 
        .i_addr_w(addrw_qmax), 
        .i_read_en(rflag_qmax),
        .i_write_en(wflag_qmax),
        .i_data(data_in_action),
        .o_data(data_out_action)); 

endmodule