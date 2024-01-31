import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles

import random
import numpy as np



def create_bw_test_image(image_size=(10, 10), square_size=4):
    # Create a simple black and white image
    image = np.ones(image_size, dtype=np.uint8)  # Black background

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


    # Time parameters
    duration = 0.025  # sec
    sampling_rate = 200000  # 200 kHz i.e clock cycle 5us
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    print("Sample Num: ", len(t))

    clock = Clock(dut.clk, 1/sampling_rate, units="sec")  
    
    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))
    
    pulse_width = 10 / 1000000  # 10us

    output_image = []
    for y in range(1, height - 1):
        dut.rst.value = 1
        await RisingEdge(dut.clk)
        dut.rst.value = 0
        liney = []
        for x in range(1, width - 1):
            # Pulse train parameters
            amplitudes = np.array([input_image[x - 1, y - 1], input_image[x - 1 + 1, y - 1], input_image[x - 1, y - 1 + 1], input_image[x - 1 + 1, y - 1 + 1]]) 
            pulse_frequencies = amplitudes  * 10000 # 10kHz for 255 pixel i.e period 100 us
            
            s = 0
            dut.s.value = 0
            for t_ in t:
                dut.r00.value = int((t_ % (1 / pulse_frequencies[0])) < (pulse_width))
                dut.r01.value = int((t_ % (1 / pulse_frequencies[1])) < (pulse_width))
                dut.r10.value = int((t_ % (1 / pulse_frequencies[2])) < (pulse_width))
                dut.r11.value = int((t_ % (1 / pulse_frequencies[3])) < (pulse_width))
                dut.sel.value = 1 if random.uniform(0, 1) < 0.5 else 0
                await RisingEdge(dut.clk)
                s += dut.s.value

            print(s)
            s = s/len(t)
            liney.append(s)
        output_image.append(liney)
    
    with open("/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/out/out_edge_detect_deter.txt", "w") as f:
        print((output_image), file=f)

'''
2. Pytest Setup
'''

from cocotb_test.simulator import run
import pytest
import glob


def test_edge_detector_deterministic():

    run(
        verilog_sources=glob.glob('hdl/*'),
        toplevel="edge_detector_deterministic",    # top level HDL
        
        module="test_edge_detector_deterministic", # name of the file that contains @cocotb.test() -- this file
        simulator="verilator"
        # parameters=parameters,
        # extra_env=parameters,
        # sim_build="sim_build/sc_add/" + ",".join((f"{key}={value}" for key, value in parameters.items())),
    )