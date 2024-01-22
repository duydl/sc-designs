import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles

import random
import numpy as np



def create_bw_test_image(image_size=(100, 100), square_size=40):
    # Create a simple black and white image
    image = np.zeros(image_size, dtype=np.uint8)  # Black background

    # Draw a white square in the center
    start_x = (image_size[0] - square_size) // 2
    start_y = (image_size[1] - square_size) // 2
    image[start_x:start_x+square_size, start_y:start_y+square_size] = 255  # White square

    return image


'''
1. Testbench
'''

@cocotb.test()
async def edge_detector(dut):

    input_image = create_bw_test_image() / 255
    width, height = input_image.shape


    # Create a 10us period clock on port clk
    clock = Clock(dut.clk, 10, units="us")  
    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))
    
    await RisingEdge(dut.clk)
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0

    output_image = []
    for y in range(1, height - 1):
        liney = []
        for x in range(1, width - 1):
            N = 1024
            s = 0
            for _ in range(N):
                dut.r00.value = 1 if random.uniform(0, 1) < input_image[x - 1, y - 1] else 0
                dut.r01.value = 1 if random.uniform(0, 1) < input_image[x - 1 + 1, y - 1] else 0
                dut.r10.value = 1 if random.uniform(0, 1) < input_image[x - 1, y - 1 + 1] else 0
                dut.r11.value = 1 if random.uniform(0, 1) < input_image[x - 1 + 1, y - 1 + 1] else 0
                dut.sel.value = 1 if random.uniform(0, 1) < 0.5 else 0
                
                await RisingEdge(dut.clk)
                s += dut.s.value

            s = s/N
            liney.append(s)
        output_image.append(liney)
    
    with open("/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/out_edge_detect.txt", "w") as f:
        print((output_image), file=f)

'''
2. Pytest Setup
'''

# from cocotb_test.simulator import run
# import glob


# def test_edge_detector():

#     run(
#         verilog_sources=glob.glob('hdl/*'),
#         toplevel="edge_detector",    # top level HDL
        
#         module="test_edge_detector", # name of the file that contains @cocotb.test() -- this file
#         simulator="icarus"

#         # parameters=parameters,
#         # extra_env=parameters,
#         # sim_build="sim_build/sc_add/" + ",".join((f"{key}={value}" for key, value in parameters.items())),
#     )



# from pathlib import Path
# import os, sys, glob
# from cocotb.runner import get_runner

# def test_():

#     verilog_sources = glob.glob('hdl/*') 
#     print(verilog_sources)
    
#     sim = os.getenv("SIM", "icarus")

#     runner = get_runner(sim)
#     runner.build(
#         verilog_sources=verilog_sources,
#         hdl_toplevel="edge_detector",
#         always=True,
#     )
#     runner.test(
#         hdl_toplevel="edge_detector", test_module="test_edge_detector"
#     )


# if __name__ == "__main__":
#     test_()