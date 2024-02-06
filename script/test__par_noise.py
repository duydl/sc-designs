import cocotb
from cocotb.clock import Clock
from cocotb.triggers import  RisingEdge

import numpy as np

from PIL import Image

m, n = 256, 256
# Shape of test image 
# Params for circuit
# circuit built before image read

def load_image(file_path):
    # Load the image using Pillow (PIL)
    img = Image.open(file_path)
    
    # Convert the image to a NumPy array
    img_array = np.array(img )
    
    return img_array

def create_bw_test_image(image_size, square_size=4):
    # Create a simple black and white image
    image = np.ones(image_size, dtype=np.uint8)  # Black background
    
    # Draw a white square in the center
    start_x = (image_size[0] - square_size) // 2
    start_y = (image_size[1] - square_size) // 2
    image[start_x:start_x+square_size, start_y:start_y+square_size] = 255  # White square
    
    ## The following will not work
    # global m,n 
    # m,n = image_size
    
    from PIL import Image
    test_img = Image.fromarray(image.astype('uint8'))
    test_img.save("/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/script/out/test_verilog.png")
    
    return image

'''
1. Testbench
'''

@cocotb.test()
async def tb(dut):
    
    # input_image = create_bw_test_image((m,n)) 
    input_image = load_image("/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/script/images/einstein_noisy.png")
    print("Shape: ", (m,n))

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
        bits_input = t_ < input_image_flatten

        dut.in_bits.value = bits_input.astype(int).tolist()
        
        await RisingEdge(dut.clk)
        
        out = list(dut.out_bits.value)
        output_image += np.array([int(bit) for bit in out])
        
    output_image = output_image.reshape((m),(n))
    
    with open("/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/script/out/out_noise_reduce_deter.txt", "w")  as f:
        print((output_image.tolist()), file=f)
        
    from PIL import Image
    edge_img = Image.fromarray(output_image.astype('uint8'))
    edge_img.save("/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/script/out/einstein_verilog.png")
    
    import os
    current_directory = os.getcwd()
    print("Current working directory:", current_directory)


'''
2. Pytest Setup
'''

import glob, os
from cocotb.runner import get_runner

def test_():

    verilog_sources = [
        "src/apps/sc_2_par_noise.sv", 
        "src/apps/sc_2_median_3x3.sv", 
        ]
    
    sim = os.getenv("SIM", "icarus")

    runner = get_runner(sim)
    
    # runner.flags = ["-sv"]
    
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="sc_par_noise",
        always=True, # always recompile
        parameters={"m":m, "n":n}
    )
    runner.test(
        hdl_toplevel="sc_par_noise", 
        test_module="test__par_noise", # this file name,
    )

if __name__ == "__main__":
    test_()