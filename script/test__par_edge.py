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
    print("Shape: ", input_image.shape)

    clock = Clock(dut.clk, 10, units="ns")  
    
    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    input_image_flatten  = input_image.flatten()  
    
    output_image = np.zeros(m*n)
    
    dut.out_bits.value = output_image.astype(int).tolist()
    
    for t_ in range(256):
        print(t_)
        bits_input = t_ < input_image_flatten
        # print("test input", pixels_input.astype(int).tolist())
        # print(bin(np.packbits(pixels_input.astype(int))))
        dut.in_bits.value = bits_input.astype(int).tolist()
        dut.sel.value = t_ % 2
        await RisingEdge(dut.clk)
        out = list(dut.result.value)
        # print("test output",   output, len(  output))
        # print("test output 2", s, len(s))
        # print(np.array([int(bit) for bit in output]))
        output_image += np.array([int(bit) for bit in out])
        
    output_image = output_image.reshape((m),(n))
    
    with open("/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/out/out_edge_detect_deter.txt", "w")  as f:
        print((output_image.tolist()), file=f)
        
    from PIL import Image
    edge_img = Image.fromarray(output_image.astype('uint8'))

    # Save the edge-detected image
    edge_img.save("cameraman_verilog.png")


'''
2. Pytest Setup
'''

import glob, os
from cocotb.runner import get_runner

def test_():

    verilog_sources = [
        "src/apps/sc_1_par_edge.sv", 
        "src/apps/sc_1_robert_op_xor.sv", 
        "src/components/mux_2_1.sv"
        ]
    
    sim = os.getenv("SIM", "icarus")

    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="sc_par_edge",
        always=True,
        parameters={"m":m, "n":n}
    )
    runner.test(
        hdl_toplevel="sc_par_edge", 
        test_module="test__par_edge", # this file name,
    )

if __name__ == "__main__":
    test_()