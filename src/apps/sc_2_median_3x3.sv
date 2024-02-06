`timescale 1us/1ns

module cas(
    input wire ai,
    input wire bi,
    output wire ao,
    output wire bo
);

    assign ao = ai & bi;
    assign bo = ai | bi;
endmodule

module sc_median_3x3(
    input logic window [0:8],
    output logic result
    // output wire result
);

// Layer 1: Sort each pair
wire layer1 [0:8];
assign layer1[8] = window[8];
genvar k;
generate
for (k = 0; k < 8; k = k + 2) begin
cas cas_inst (
    .ai(window[k]),  // Connect input ai to window[k]
    .bi(window[k + 1]),  // Connect input bi to window[k + 1]
    .ao(layer1[k]),  // Connect output ao to window[k]
    .bo(layer1[k + 1])  // Connect output bo to window[k + 1]
);
end
endgenerate

// Layer 2: Merge pairs
wire layer2 [0:8];
assign layer2[8] = layer1[8];
cas cas_inst_21 (.ai(layer1[0]), .bi(layer1[2]), .ao(layer2[0]), .bo(layer2[2]));
cas cas_inst_22 (.ai(layer1[1]), .bi(layer1[3]), .ao(layer2[1]), .bo(layer2[3]));
cas cas_inst_23 (.ai(layer1[4]), .bi(layer1[6]), .ao(layer2[4]), .bo(layer2[6]));
cas cas_inst_24 (.ai(layer1[5]), .bi(layer1[7]), .ao(layer2[5]), .bo(layer2[7]));

// Layer 3
wire layer3 [0:8];  
assign layer3[8] = layer2[8];
cas cas_inst_31 (.ai(layer2[1]), .bi(layer2[2]), .ao(layer3[1]), .bo(layer3[2]));
cas cas_inst_32 (.ai(layer2[5]), .bi(layer2[6]), .ao(layer3[5]), .bo(layer3[6]));
cas cas_inst_33 (.ai(layer2[0]), .bi(layer2[4]), .ao(layer3[0]), .bo(layer3[4]));
cas cas_inst_34 (.ai(layer2[3]), .bi(layer2[7]), .ao(layer3[3]), .bo(layer3[7]));

// Layer 4
wire layer4 [0:8];  
assign layer4[0] = layer3[0];
assign layer4[3] = layer3[3];
assign layer4[4] = layer3[4];
assign layer4[7] = layer3[7];
assign layer4[8] = layer3[8];
cas cas_inst_41 (.ai(layer3[1]), .bi(layer3[5]), .ao(layer4[1]), .bo(layer4[5]));
cas cas_inst_42 (.ai(layer3[2]), .bi(layer3[6]), .ao(layer4[2]), .bo(layer4[6]));

// Layer 5
wire layer5 [0:8];  
assign layer5[0] = layer4[0];
assign layer5[1] = layer4[1];
assign layer5[6] = layer4[6];
assign layer5[7] = layer4[7];
assign layer5[8] = layer4[8];
assign layer5[8] = layer4[8];
cas cas_inst_51 (.ai(layer4[2]), .bi(layer4[4]), .ao(layer5[2]), .bo(layer5[4]));
cas cas_inst_52 (.ai(layer4[3]), .bi(layer4[5]), .ao(layer5[3]), .bo(layer5[5]));

// Layer 6
wire layer6 [0:8];  
assign layer6[0] = layer5[0];
assign layer6[1] = layer5[1];
assign layer6[2] = layer5[2];
assign layer6[5] = layer5[5];
assign layer6[6] = layer5[6];
assign layer6[7] = layer5[7];
assign layer6[8] = layer5[8];
cas cas_inst_6 (.ai(layer5[3]), .bi(layer5[4]), .ao(layer6[3]), .bo(layer6[4]));

// Layer 7: Include 9th element to find median
wire layer7 [0:8]; 
assign layer7[0] = layer6[0];
assign layer7[1] = layer6[1];
assign layer7[2] = layer6[2];
assign layer7[4] = layer6[4];
assign layer7[5] = layer6[5];
assign layer7[6] = layer6[6];
assign layer7[7] = layer6[7];
cas cas_inst_7 (.ai(layer6[3]), .bi(layer6[8]), .ao(layer7[3]), .bo(layer7[8]));

// Result
wire result_ [0:8]; 
assign result_[0] = layer7[0];
assign result_[1] = layer7[1];
assign result_[2] = layer7[2];
assign result_[3] = layer7[3];
assign result_[5] = layer7[5];
assign result_[6] = layer7[6];
assign result_[7] = layer7[7];
cas cas_inst_result (.ai(layer7[4]), .bi(layer7[8]), .ao(result_[4]), .bo(result_[8]));

assign result = result_[4];

endmodule