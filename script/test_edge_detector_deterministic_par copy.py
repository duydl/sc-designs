# import cocotb
# from cocotb.clock import Clock
# from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles
# from cocotb.binary import BinaryValue

# import random
# import numpy as np

# from PIL import Image

# def create_bw_test_image(image_size=(10, 10), square_size=4):
#     # Create a simple black and white image
#     image = np.ones(image_size, dtype=np.uint8)  # Black background

#     # Draw a white square in the center
#     start_x = (image_size[0] - square_size) // 2
#     start_y = (image_size[1] - square_size) // 2
#     image[start_x:start_x+square_size, start_y:start_y+square_size] = 255  # White square

#     return image

# def load_image(file_path, target_size):
#     # Load the image using Pillow (PIL)
#     img = Image.open(file_path)

#     img_resized = img.resize(target_size)
    
#     # Convert the image to a NumPy array
#     img_array = np.array(img_resized )
    
#     grayscale_image = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

#     return grayscale_image

# '''
# 1. Testbench
# '''
# m = 32
# n = 32
# @cocotb.test()
# async def edge_detector(dut):
    
#     # input_image = create_bw_test_image((m,n)) / 255
#     input_image = load_image("/home/ubuntu20_1/WSL_dev_projs/verilog/sc_designs/Lenna_(test_image).png", (m,n)) / 255
#     width, height = input_image.shape
#     print("shape", input_image.shape)

#     # Time parameters
#     duration = 0.025  # sec
#     sampling_rate = 200000  # 200 kHz i.e clock cycle 5us
#     t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
#     print("Sample Num: ", len(t))

#     clock = Clock(dut.clk, 1/sampling_rate, units="sec")  
    
#     # Start the clock. Start it low to avoid issues on the first RisingEdge
#     cocotb.start_soon(clock.start(start_high=False))
    
#     pulse_width = 10 / 1000000  # 10us
#     dut.rst.value = 1
#     await RisingEdge(dut.clk)
#     dut.rst.value = 0
#     liney = []
#     output_image = []
    
#     pulse_frequencies = input_image.flatten()  * 10000 # 10kHz for 255 pixel i.e period 100 us 
#     s = np.zeros((m-1)*(n-1))
#     dut.s.value = s.astype(int).tolist()
#     for t_ in t:
#         pixels_input = (t_ % (1 / pulse_frequencies)) < pulse_width
#         # print("test input", pixels_input.astype(int).tolist())
#         # print(bin(np.packbits(pixels_input.astype(int))))
#         dut.pixels.value = pixels_input.astype(int).tolist()
#         dut.sel.value = 1 if random.uniform(0, 1) < 0.5 else 0
#         await RisingEdge(dut.clk)
#         output = list(dut.s.value)
#         # print("test output",   output, len(  output))
#         # print("test output 2", s, len(s))
#         # print(np.array([int(bit) for bit in output]))
#         s += np.array([int(bit) for bit in output])
        
#     s = s/len(t)
#     # s = s/2
#     s = s.reshape((m-1),(n-1)).tolist()
    
#     with open("/home/ubuntu20_1/WSL_dev_projs/verilog/sc_designs/out/out_edge_detect_deter_2.txt", "w") as f:
#         print((s), file=f)


# '''
# 2. Pytest Setup
# '''

# from cocotb_test.simulator import run
# import pytest
# import glob


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