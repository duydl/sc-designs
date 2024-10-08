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
async def sc_tanh_tb(dut):
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
            # Debug
            print(_)
            print(f"y = {dut.y.value}")
            print(f"x = {dut.x.value}")
            print(f"state = {dut.current_state.value}")
            print(f"nextstate = {dut.next_state.value}")
            y += dut.y.value
           
        y_range.append(y/N)

    print(x_range)
    print(y_range)
    with open("/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/out_tanh.txt", "w") as f:
        print(list(x_range), file=f)
        print(list(y_range), file=f)



'''
2. Pytest Setup
'''

from cocotb_test.simulator import run
import glob


def test_tanh():

    run(
        verilog_sources=glob.glob('hdl/*'),
        toplevel="sc_tanh",    # top level HDL
        
        module="test_tanh", # name of the file that contains @cocotb.test() -- this file
        simulator="icarus"

        # parameters=parameters,
        # extra_env=parameters,
        # sim_build="sim_build/sc_add/" + ",".join((f"{key}={value}" for key, value in parameters.items())),
    )