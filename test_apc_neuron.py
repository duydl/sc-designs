import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles

import random

'''
1. Testbench
'''

@cocotb.test()
async def sc_apc_neuron(dut):


    # Set initial input value to prevent it from floating
    dut.din.value = 4
    dut.weight.value = 7
    
    clock = Clock(dut.clk, 10, units="us")  # Create a 10us period clock on port clk
    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))
    
    await RisingEdge(dut.clk)

    
    for i in range(10): # 10 experiments
        N = 10
        output = 0
        dut.reset.value = 1
        await RisingEdge(dut.clk)
        dut.reset.value = 0
        for _ in range(N):
            # Generate random input streams with probabilities -0.4 and 0.6
            # dut.a.value = 1 if random.uniform(0, 1) < pa else 0
            # dut.b.value = 1 if random.uniform(0, 1) < pb else 0
            # dut.select.value = random.randint(0, 1)  # Set select to 0.5
            
            await RisingEdge(dut.clk)
            
            
            # Calculate expected output based on select
            output += dut.dout.value
            print(dut.current_state.value)
            print(dut.count.value)
           
            
        pc = output / N
        # c = 2 * pc - 1
        # assert output == (a + b)/2.0, f"Failed on the {i}th experiment. Got {output}, expected {(a + b)/2.0}"
        print(f"Test {i}:")
        print(f" Prob Output: {pc}")



'''
2. Pytest Setup
'''

from cocotb_test.simulator import run
import pytest
import glob


def test_add():

    run(
        verilog_sources=glob.glob('hdl/*'),
        toplevel="sc_apc_neuron",    # top level HDL
        
        module="test_apc_neuron", # name of the file that contains @cocotb.test() -- this file
        simulator="icarus"

        # parameters=parameters,
        # extra_env=parameters,
        # sim_build="sim_build/sc_add/" + ",".join((f"{key}={value}" for key, value in parameters.items())),
    )