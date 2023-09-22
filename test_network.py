import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import random
import numpy as np


def list_to_binary(arr):
    return int(''.join(map(str, arr)), 2)


'''
1. Testbench
'''

@cocotb.test()
async def sc_mux_neuron_tb(dut):



    func = np.vectorize(lambda x: int(random.random() < x))

# Create a model with the same architecture as the original model
    model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

    # Load the saved weights into the new model
    model.load_weights('/home/ubuntu20_1/WSL_dev_projs/verilog/sc_designs/mnist_model_weights.h5')
    layer_weights = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()
            if weights:
                layer_weights.append((1 + weights[0])/2)

    (_, _), (test_images, test_labels) = mnist.load_data()
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    clock = Clock(dut.clk, 10)  

    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0

    
    for i, test_image in enumerate(test_images[0:1]): # 10 experiments
        
        N = 1024
        output = 0

        for _ in range(N):
            print(_)

            dut.din.value = list_to_binary( func(test_image.flatten()))

            # dut.sel1.value = [random.randint(0,784) for i in range(128)]
            # dut.sel2.value = [random.randint(0,128) for i in range(10)]
            

            dut.weight_0.value = [list_to_binary(x) for x in func(layer_weights[0].T)]
            dut.weight_1.value = [list_to_binary(x) for x in func(layer_weights[1].T)]
            dut.weight_0.value = [1 for x in func(layer_weights[0].T)]
            dut.weight_1.value = [1 for x in func(layer_weights[1].T)]

            
            # print(len([list_to_binary(x) for x in func(layer_weights[1].T)]), [list_to_binary(x) for x in func(layer_weights[1].T)])

            await RisingEdge(dut.clk)

            # print(len(dut.weight_1.value) ,dut.weight_1.value)
            # print("input," , dut.din.value)
            print("output", dut.dout.value)
            # print("weight", dut.weight_1.value[6])
            print("output layer", dut.layer1_output.value)
            print("output layer", dut.layer1_output.value)

            # Calculate expected output based on select
            # if "x" in str(dut.dout.value):
                
            #     count += 1
            # else:
            #     output += dut.dout.value

            output += np.array((list(str(dut.dout.value)))).astype(int)

        pc = output / N


        print(f"Test {i}:")
        print(f"Expected Prob: {test_labels[i]} \t Prob Output: {pc}")



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