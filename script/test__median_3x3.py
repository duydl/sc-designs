import cocotb
from cocotb.triggers import  Edge, Timer

'''
1. Testbench
'''

@cocotb.test()
async def tb(dut):

    window = [0, 1, 0, 1, 0, 1, 0, 1, 0]
    dut.window.value = window
    
    # Wait for an edge of input 
    # Not need in this test 1ns Timer is enough
    await Edge(dut.result)  
    
    # Without time, circuit output "x x"
    await Timer(1, "ns")
    
    print(dut.window.value)
    print(dut.layer1.value)
    print(dut.layer2.value)
    print(dut.layer3.value)
    print(dut.layer4.value)
    print(dut.layer5.value)
    print(dut.layer6.value)
    print(dut.layer7.value)
    print(dut.result_.value)
    print(dut.result.value)

'''
2. Pytest Setup
'''

import glob, os
from cocotb.runner import get_runner

def test_():

    verilog_sources = [
        "src/apps/sc_2_median_3x3.sv", 
        ]
    
    sim = os.getenv("SIM", "icarus")

    runner = get_runner(sim)
    
    # runner.flags = ["-sv"]
    
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="sc_median_3x3",
        always=True,
    )
    runner.test(
        hdl_toplevel="sc_median_3x3", 
        test_module="test__median_3x3", # this file name,
    )

if __name__ == "__main__":
    test_()