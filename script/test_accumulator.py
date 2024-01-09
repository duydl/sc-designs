import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles

import random

'''
1. Testbench
'''

@cocotb.test()
async def sc_accumulator(dut):


    # Set initial input value to prevent it from floating
    dut.data_in.value = 11
    
    clock = Clock(dut.clk, 10, units="us")  # Create a 10us period clock on port clk
    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))
    
    await RisingEdge(dut.clk)

    
    for i in range(10): # 10 experiments
        N = 4
        
        for _ in range(N):
            # dut.count.value = 0
            print("datain", dut.data_in.value)
            dut.data_in.value = _ + 24
            await RisingEdge(dut.clk)
            print("count", dut.count.value)
            

        print(f"Test {i}:")




'''
2. Pytest Setup
'''

from cocotb_test.simulator import run
import glob


def test_add():

    run(
        verilog_sources=glob.glob('hdl/*'),
        toplevel="accumulator",    # top level HDL
        
        module="test_accumulator", # name of the file that contains @cocotb.test() -- this file
        simulator="icarus"

        # parameters=parameters,
        # extra_env=parameters,
        # sim_build="sim_build/sc_add/" + ",".join((f"{key}={value}" for key, value in parameters.items())),
    )