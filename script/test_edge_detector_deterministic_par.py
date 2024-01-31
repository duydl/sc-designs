import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles
from cocotb.binary import BinaryValue

import random
import numpy as np

from PIL import Image

def create_bw_test_image(image_size=(10, 10), square_size=4):
    # Create a simple black and white image
    image = np.ones(image_size, dtype=np.uint8)  # Black background

    # Draw a white square in the center
    start_x = (image_size[0] - square_size) // 2
    start_y = (image_size[1] - square_size) // 2
    image[start_x:start_x+square_size, start_y:start_y+square_size] = 255  # White square

    return image

def load_image(file_path, target_size):
    # Load the image using Pillow (PIL)
    img = Image.open(file_path)
    
    # Convert the image to a NumPy array
    img_array = np.array(img )
    


    return img_array

'''
1. Testbench
'''
m = 256
n = 256
@cocotb.test()
async def edge_detector(dut):
    
    # input_image = create_bw_test_image((m,n)) 
    input_image = load_image("/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/submodules/sc-simulation/cameraman.png", (m,n))
    print(input_image)
    print("shape", input_image.shape)

    # Time parameters
     # sec
    sampling_rate = 200000  # 200 kHz i.e clock cycle 5us
    duration = 256/ sampling_rate

    clock = Clock(dut.clk, 10, units="ns")  
    
    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    liney = []
    output_image = []
    
    input_image_flatten  = input_image.flatten()  # 10kHz for 255 pixel i.e period 100 us 
    s = np.zeros(m*n)
    dut.s.value = s.astype(int).tolist()
    for t_ in range(256):
        print(t_)
        pixels_input = t_ < input_image_flatten
        # print("test input", pixels_input.astype(int).tolist())
        # print(bin(np.packbits(pixels_input.astype(int))))
        dut.pixels.value = pixels_input.astype(int).tolist()
        dut.sel.value = t_ % 2
        await RisingEdge(dut.clk)
        output = list(dut.s.value)
        # print("test output",   output, len(  output))
        # print("test output 2", s, len(s))
        # print(np.array([int(bit) for bit in output]))
        s += np.array([int(bit) for bit in output])
        
    # s = s/2
    s = s.reshape((m),(n))
    
    with open("/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/out/out_edge_detect_deter.txt", "w")  as f:
        print((s.tolist()), file=f)
        
    from PIL import Image
    edge_img = Image.fromarray(s.astype('uint8'))

    # Save the edge-detected image
    edge_img.save("cameraman_verilog.png")


'''
2. Pytest Setup
'''

from cocotb_test.simulator import run
import pytest
import glob


# def test_edge_detector_deterministic():

#     run(
#         verilog_sources=glob.glob('hdl/*'),
#         toplevel="image_processor",    # top level HDL
        
#         module="test_edge_detector_deterministic_par", # name of the file that contains @cocotb.test() -- this file
#         simulator="icarus",
#         parameters={"m":m, "n":n}
#         # parameters=parameters,
#         # extra_env=parameters,
#         # sim_build="sim_build/sc_add/" + ",".join((f"{key}={value}" for key, value in parameters.items())),
#     )


import glob, os
from cocotb.runner import get_runner

def test_edge_detector():

    verilog_sources = ["hdl/edge_detector_par.sv", "hdl/edge_detector_deterministic_xor_abs.sv", "hdl/sc_add.sv"]
    # proj_path = Path(__file__).resolve().parent.parent
    
    # verilog_sources = [proj_path / "hdl" / "sc_abs.sv"]
    
    sim = os.getenv("SIM", "icarus")

    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="image_processor",
        always=True,
        parameters={"m":m, "n":n}
    )
    runner.test(
        hdl_toplevel="image_processor", test_module="test_edge_detector_deterministic_par", #file name,
        
        
        
    )


if __name__ == "__main__":
    test_edge_detector()