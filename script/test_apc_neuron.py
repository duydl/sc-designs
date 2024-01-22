import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles

import random
import numpy as np

'''
1. Testbench
'''

@cocotb.test()
async def sc_apc_neuron(dut):

    x_range = np.linspace(0,1,81)
    # Set initial input value to prevent it from floating
    dut.din.value = 4
    dut.weight.value = 255
    
    clock = Clock(dut.clk, 10, units="us")  # Create a 10us period clock on port clk
    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))
    
    await RisingEdge(dut.clk)

    y_range = []
    for x in x_range: 
        # x = 0.8
        N = 256
        y = 0
        dut.reset.value = 1
        await RisingEdge(dut.clk)
        dut.reset.value = 0
        for _ in range(N):
            print(_)
            print("mem", dut.mem1.value)
            
            dut.din.value = 1*int(random.random() < x) + 2*int(random.random() < x) + 4*int(random.random() < x) + 8*int(random.random() < x) + 16*int(random.random() < x)  + 32*int(random.random() < x) + 64*int(random.random() < x)  + 128*int(random.random() < x) 
            
            await RisingEdge(dut.clk)
            
            # Calculate expected output based on select
            y += dut.dout.value
            # if _ % 100 == 0:
            
            
            print("count", dut.count.value)
            print("current state", dut.current_state.value)
            # print("N/2", dut.N.value)
            
            
        y_range.append(y/N)
    with open("/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/out_apc_neuron.txt", "w") as f:
        print(list(x_range), file=f)
        print(list(y_range), file=f)



'''
2. Pytest Setup
'''

from cocotb_test.simulator import run
import glob


def test_apc_neuron():

    run(
        verilog_sources=glob.glob('hdl/*'),
        toplevel="sc_apc_neuron",    # top level HDL
        module="test_apc_neuron", # name of the file that contains @cocotb.test() -- this file
        simulator="icarus"

        # parameters=parameters,
        # extra_env=parameters,
        # sim_build="sim_build/sc_add/" + ",".join((f"{key}={value}" for key, value in parameters.items())),
    )