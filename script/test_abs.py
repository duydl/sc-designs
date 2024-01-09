import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles

import random
import numpy as np
import matplotlib.pyplot as plt
import os 

'''
1. Testbench
'''

@cocotb.test()
async def sc_abs_tb(dut):
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
        print(y/N)
        y_range.append(y/N)

    with open(os.getcwd() + "/../out/out_abs.txt", "w") as f:
        print(list(x_range), file=f)
        print(list(y_range), file=f)



'''
2. Pytest Setup
'''

# from cocotb_test.simulator import run
# import glob

# def test_abs():

#     run(
#         verilog_sources=glob.glob('hdl/*'),
#         toplevel="sc_abs",    # top level HDL
#         module="test_abs", # name of the file that contains @cocotb.test() -- this file
#         simulator="icarus"
#     )


import glob
from cocotb.runner import get_runner

def test_abs():

    verilog_sources = glob.glob('hdl/*') 
    # proj_path = Path(__file__).resolve().parent.parent
    
    # verilog_sources = [proj_path / "hdl" / "sc_abs.sv"]
    
    sim = os.getenv("SIM", "icarus")

    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="sc_abs",
        always=True,
    )
    runner.test(
        hdl_toplevel="sc_abs", test_module="test_abs"
    )


if __name__ == "__main__":
    test_abs()