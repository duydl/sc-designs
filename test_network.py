import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles

import random


'''
1. Testbench
'''

@cocotb.test()
async def sc_mux_neuron_tb(dut):
    probabilities_weight = [[0.1] * 4] * 10

    probabilities_in = [0.1] * 100


    # Set initial input value to prevent it from floating
    dut.din.value = random.randint(0, 7)
    dut.sel1.value = random.randint(0, 2)

    clock = Clock(dut.clk, 10)  # Create a 10us period clock on port clk
    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0

    
    for i in range(3): # 10 experiments
        N = 1024
        output = 0
        for _ in range(N):

            binary_string = "".join(str(int(random.random() < p)) for p in probabilities_in)

            # Convert the binary string to an integer
            binary_number = int(binary_string, 2)

            dut.din.value = binary_number
            dut.sel1.value = random.randint(0, 7) 
            
            await RisingEdge(dut.clk)

            # Calculate expected output based on select
            output += dut.dout.value

        pc = output / N


        print(f"Test {i}:")
        # print(f"Expected Prob: {sum(probabilities) / len(probabilities)} \t Prob Output: {pc}")



'''
2. Pytest Setup
'''

from cocotb_test.simulator import run
import pytest
import glob


def test_mux_neuron():

    run(
        verilog_sources=glob.glob('hdl/*'),
        toplevel="sc_mnist_network",    # top level HDL
        
        module="test_network", # name of the file that contains @cocotb.test() -- this file
        simulator="icarus"

        # parameters=parameters,
        # extra_env=parameters,
        # sim_build="sim_build/sc_add/" + ",".join((f"{key}={value}" for key, value in parameters.items())),
    )