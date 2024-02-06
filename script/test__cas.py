import cocotb
from cocotb.clock import Clock
from cocotb.triggers import  RisingEdge, FallingEdge, Edge, Timer

import numpy as np

from PIL import Image

m, n = 256, 256
# Shape of test image 
# Params for circuit
# circuit built before image read

'''
1. Testbench
'''

@cocotb.test()
async def tb(dut):
    
    dut.ai.value = 1
    dut.bi.value = 0
    print("Initial values: ao =", dut.ao.value, "bo =", dut.bo.value)
    
    # Wait for an edge of input 
    await Edge(dut.ao)  
    await Edge(dut.bo)
    
    # Without time, circuit output "x x"
    await Timer(1, "ns")

    assert dut.ao.value == 0, "a should be 1"
    assert dut.bo.value == 1, "b should be updated to 1"
    print("After values: ao =", dut.ao.value, "bo =", dut.bo.value)

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
        hdl_toplevel="cas",
        always=True,
    )
    runner.test(
        hdl_toplevel="cas", 
        test_module="sc_cas_AND_OR.sv", # this file name,
    )

if __name__ == "__main__":
    test_()