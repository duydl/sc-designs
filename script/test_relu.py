import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles

import random
import numpy as np
import matplotlib.pyplot as plt

'''
1. Testbench
'''

@cocotb.test()
async def sc_relu_tb(dut):
    x_range = np.linspace(0,1,41)

    # Set initial input value to prevent it from floating
    dut.x.value = 0
    clock = Clock(dut.clk, 10, units="us")  # Create a 10us period clock on port clk
    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))
    
    await RisingEdge(dut.clk)

    y_range = []
    for x in x_range: 
        N = 1024
        y = 0
        dut.reset.value = 1
        await RisingEdge(dut.clk)
        dut.reset.value = 0

        for _ in range(N):
            # Generate random input streams 
            dut.x.value = 1 if random.uniform(0, 1) < x else 0
            
            await RisingEdge(dut.clk)
            # print(_)
            # print(f"y = {dut.y.value}")
            # print(f"x = {dut.x.value}")
            # print(f"state = {dut.current_state.value}")
            # print(f"nextstate = {dut.next_state.value}")
            y += dut.y.value
           
        y_range.append(y/N)

    with open("/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/out/out_relu.txt", "w") as f:
        print(list(x_range), file=f)
        print(list(y_range), file=f)



'''
2. Pytest Setup
'''

from cocotb_test.simulator import run
import glob


def test_relu():

    run(
        verilog_sources=glob.glob('submodules/UnarySim/hw/kernel/old/relu/relu.sv'),
        toplevel="relu", # top level HDL
        module="test_relu", # name of file contains @cocotb.test
        simulator="icarus"
    )
   

# def test_relu():

#     run(
#         verilog_sources=glob.glob('hdl/sc_relu_2.sv'),
#         toplevel="sc_relu_2", # top level HDL
#         module="test_relu", # name of file contains @cocotb.test
#         simulator="icarus"
#     ) 
    

# def test_abs():

#     run(
#         verilog_sources=glob.glob('submodules/UnarySim/hw/kernel/abs/Abs.sv'),
#         toplevel="Abs", # top level HDL
#         module="test_relu", # name of file contains @cocotb.test
#         simulator="icarus"
#     )