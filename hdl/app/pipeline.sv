`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09/02/2019 10:53:28 AM
// Design Name: 
// Module Name: pipeline
// Project Name: 
// Target Devices:  
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


//store tables in BRAMs
//width depends on range of q value, depth depends on number of states times num of actions

//The 4-stage pipeline
//inputs: action
module pipeline  #(parameter ADDR_WIDTH = 8, DATA_WIDTH = 32, DEPTH = 16) ( input clk,input rst, input[1:0] action, output reg[47:0] sum);

    //used in stage 1
    reg[DATA_WIDTH-1:0] q; //q value
    reg[DATA_WIDTH-1:0] r; //reward
    reg[DATA_WIDTH-1:0] q1; //q value
    reg[DATA_WIDTH-1:0] r1; //reward
    reg[DATA_WIDTH-1:0] qmax;
    
    reg[5:0] s; //2^6 possible states (8x8 (x,y) grid, s[5:3]s -> x, s[2:0] -> y)
    reg[7:0] alpha;
    reg[7:0] oneminusa; //1-alpha
    reg[7:0] gamma;
    reg[15:0] ag; //alpha*gamma

    //propagate for qmax writing address
    reg[5:0] current_s ;
    reg[5:0] current_s1 ;
    reg[5:0] current_s2 ;
    reg[5:0] current_s3 ;
        
    //propagate for q writing address
    reg[1:0] current_a ;
    reg[1:0] current_a1 ;
    reg[1:0] current_a2 ;
    reg[1:0] current_a3 ;
    
    reg[2:0] sx ; // s[5:3]s -> x, 
    reg[2:0] sy ; // s[2:0]   -> y)
    reg[5:0] nexts; //next state for state transition

    //used in stage 2
    
    //used in stage 3
    //reg [23:0] sum;
    wire  agvalid;
    //used in stage 1 and 4
    //used for q table reading & writing 
    reg [ADDR_WIDTH-1:0] addrr_q;  
    reg [ADDR_WIDTH-1:0] addrw_q;
    //reg [7:0] addrr_q_tmp;
    //reg [7:0] addr_r_tmp;
    reg rflag_q; //0 or 1
    reg wflag_q; //0 or 1
    reg [DATA_WIDTH-1:0] data_in_q;
    wire [DATA_WIDTH-1:0] data_out_q;

    //used for qmax table reading & writing
    reg [5:0] addrr_qmax;
    reg [5:0] addrw_qmax;
    reg rflag_qmax; //0 or 1
    reg wflag_qmax; //0 or 1
    reg [DATA_WIDTH-1:0] data_in_qmax;
    wire [DATA_WIDTH-1:0] data_out_qmax;

    //used for r table reading
    reg [ADDR_WIDTH-1:0] addr_r;
    reg rflag_r; //0 or 1
    wire [DATA_WIDTH-1:0] data_out_r;
    localparam sf = 2.0**-4.0;
    //--------------stage 1-----------------
    always @(posedge clk) begin
    //initialize state and action
        if (rst) begin
            s<=6'b000_000;
            current_s<=6'b000000;
            nexts<=6'b000000;;
            alpha<=8'b0000_0010; //0.8
            gamma<=8'b0000_0010;
        end
 
        //calculate 1-a and a*g 
        //scaling factor=2.0**-4.0 _
        ag <= alpha*gamma;
        oneminusa <= 8'b0001_0000 - alpha;        
        
        //locate next state
        sx<=s[5:3];sy<=s[2:0]; 
        if (sx==3'b000 && action==2'b00) begin //left wall 
            nexts<=s;
        end
        else if (sy==3'b000 && action==2'b01) begin //up wall
            nexts<=s;      
        end
        else if (sx==3'b111 && action==2'b10) begin //right wall
            nexts<=s;      
        end
        else if (sy==3'b111 && action==2'b11) begin //down wall
            nexts<=s;     
        end
        else begin
            case (action)
                2'b00: nexts<=s-6'b001000;//to the left by 1
                2'b01: nexts<=s-6'b000001;//to the up by 1
                2'b10: nexts<=s+6'b001000;//to the right by 1
                2'b11: nexts<=s+6'b000001;//to the down by 1
            //default:
            endcase
            //nexts<={sx,sy};
        end
        
        //get address for q and r and qmax
        addrr_q<={s,action}; 
        addr_r<={s,action};
        addrr_qmax<=nexts;
        
        //wait and transit the state
        current_s<=s;
        current_s1<=current_s;
        current_a<=action;
        current_a1<=current_a;
        s<=nexts;  
    end   
    
    
    
    //--------------stage 2-----------------
    always @(posedge clk) begin
    //locate q value from q table, save in q register
        rflag_q<=1;
        q<=data_out_q;
        q1<=q;
        
        rflag_r<=1;
        r<=data_out_r;
        r1<=r;

        //locate Qmax at next state from Qmax table
        rflag_qmax<=1;
        qmax<=data_out_qmax;
        
        current_s2<=current_s1;
        current_a2<=current_a1;
        
    end
    
    //--------------stage 3-----------------
    //always @(qmax or r or q or ag or oneminusa) begin 
    
    /*reg [23:0] sum_part1;
    reg [23:0] sum_part2;
    reg [23:0] sum_part3;
    
    always@(posedge clk)
    begin
        sum_part1 <= alpha*r1;
        sum_part2 <= oneminusa*q1;
        sum_part3 <= ag*qmax;
    end
    
    
    always @(posedge clk) begin
        //calculations of q learning function
                //adder
        sum <= sum_part1 + sum_part2 + sum_part3;
        
        current_s3<=current_s2;
        current_a3<=current_a2;
    end*/
    always @(posedge clk) begin
        //calculations of q learning function
                //adder
        sum <= alpha*r1 + oneminusa*q1 + ag*qmax;
        //sum <= alpha*r1*2**(-4) + oneminusa*q1*2**(-4) + ag*qmax*2**(-8);
        
        current_s3<=current_s2;
        current_a3<=current_a2;

    end    
    
    

    //--------------stage 4-----------------
    //always @(sum) begin
    always @(posedge clk) begin
   // if(ce) begin
        //write back to qmax table
        if (sum>q)begin
            wflag_qmax<=1;
            addrw_qmax<=current_s3;
            data_in_qmax<=sum;
        end
        //write back to q table
        wflag_q<=1;
        addrw_q<={current_s3,current_a3}; 
        data_in_q<=sum;
        //stop the pipeline if reached end state
        //if (current_s3 == 6'b111111) begin
        //    $finish;
        //end
    //end
    end
        
    qtable qt0(
        .i_clk(clk),
        .i_rst(rst),
        .i_addr_r(addrr_q), 
        .i_addr_w(addrw_q),
        .i_read_en(rflag_q), 
        .i_write_en(wflag_q),
        .i_data(data_in_q),
        .o_data(data_out_qdata_out_q)); 

    qmaxtable qmaxt0(
        .i_clk(clk),
        .i_rst(rst),
        .i_addr_r(addrr_qmax), 
        .i_addr_w(addrw_qmax), 
        .i_read_en(rflag_qmax),
        .i_write_en(wflag_qmax),
        .i_data(data_in_qmax),
        .o_data(data_out_qmax)); 

    rtable rt0(
        .i_clk(clk),
        .i_addr(addr_r), 
        .i_read(rflag_r), 
        .o_data(data_out_r));
        
 /*   floating_point_0 mult (
      .aclk(clk),                                  // input wire aclk
      .s_axis_a_tvalid(1'b0),            // input wire s_axis_a_tvalid
      .s_axis_a_tdata(alpha),              // input wire [31 : 0] s_axis_a_tdata
      .s_axis_b_tvalid(1'b0),            // input wire s_axis_b_tvalid
      .s_axis_b_tdata(gamma),              // input wire [31 : 0] s_axis_b_tdata
      .m_axis_result_tvalid(agvalid),  // output wire m_axis_result_tvalid
      .m_axis_result_tdata(ag)    // output wire [31 : 0] m_axis_result_tdata
    );*/

endmodule