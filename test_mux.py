import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles

import random

'''
1. Testbench
'''



@cocotb.test()
async def sc_mux_neuron_tb(dut):

    # Set initial input value to prevent it from floating
    for i in range(dut.N.value):
        dut.din.value[i] = random.randint(0, 1) 
    for i in range(dut.K.value):
        dut.sel.value[i] = random.randint(0, 1) 

    clock = Clock(dut.clk, 2, units="us")  # Create a 10us period clock on port clk
    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))
    
    await RisingEdge(dut.clk)

    
    for i in range(10): # 10 experiments
        N = 256
        output = 0
        for _ in range(N):
            # Generate random input streams with probabilities -0.4 and 0.6
            for i in range(dut.N.value):
                dut.din.value[i] = random.randint(0, 1) 
            for i in range(dut.K.value):
                dut.sel.value[i] = random.randint(0, 1) 


            
            await RisingEdge(dut.clk)
            print(dut.din.value)
            print(dut.sel.value)
            
            # Calculate expected output based on select
            output += dut.dout.value
           
            
        pc = output / N
        c = 2 * pc - 1
        # assert output == (a + b)/2.0, f"Failed on the {i}th experiment. Got {output}, expected {(a + b)/2.0}"
        print(f"Test {i}:")
        print(f"Expected Prob: 0.5 \t Prob Output: {pc}")
        print(f"Expected value: 0.5 \t Output: {c}")



'''
2. Pytest Setup
'''

from cocotb_test.simulator import run
import pytest
import glob


def test_mux_neuron():

    run(
        verilog_sources=glob.glob('hdl/*'),
        toplevel="sc_mux_neuron",    # top level HDL
        
        module="test_mux", # name of the file that contains @cocotb.test() -- this file
        simulator="icarus"

        # parameters=parameters,
        # extra_env=parameters,
        # sim_build="sim_build/sc_add/" + ",".join((f"{key}={value}" for key, value in parameters.items())),
    )