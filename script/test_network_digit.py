import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles


import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler  # For input normalization
from sklearn import datasets

import numpy as np
# Load the Digits dataset
# Load the Digits dataset
digits = datasets.load_digits()

# Choose two classes (e.g., classes 0 and 1) for binary classification
class_0 = 2
class_1 = 3

# Filter the dataset to only include the chosen classes
mask = (digits.target == class_0) | (digits.target == class_1)
data = digits.data[mask]
labels = digits.target[mask]
labels = [0 if label == class_0 else 1 for label in labels]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalize input data to the range [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define an MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh() 
        self.sigmoid = nn.Sigmoid()# Sigmoid activation for binary classification

        self.fc1.bias.data.fill_(0)
        self.fc2.bias.requires_grad = False
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.sigmoid(out)  # Apply sigmoid to output
        # out = self.sigmoid(out)
        return out


# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 32
output_size = 1  # Binary classification

# Initialize the model
model = MLPModel(input_size, hidden_size, output_size)
# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1, 1))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = (y_pred >= 0.5).view(-1)  # Convert probabilities to binary predictions

print(model(X_test[0:5]))

accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())
# print(f'Accuracy: {accuracy:.2f}')


model_state_dict = model.state_dict()
# print((model_state_dict['fc1.weight'].clone() + 1) / 2)
# print((model_state_dict['fc2.weight'].clone() + 1) / 2)

# fc1_bias = (model_state_dict['fc1.bias'].numpy()+ 1) / 2
# fc2_bias = (model_state_dict['fc2.bias'].numpy()+ 1) / 2
fc1_weight = (model_state_dict['fc1.weight'].numpy() + 1) / 2
fc2_weight = (model_state_dict['fc2.weight'].numpy() + 1) / 2
test_images_prob = (1 + X_test.numpy())/2

'''
1. Testbench
'''

@cocotb.test()
async def sc_network_tb(dut):
 

    
    clock = Clock(dut.clk, 10)  

    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))


    a, b = 0,-1
    result = []
    
    
    
    for i, test_image in enumerate(test_images_prob[a:b]): # 10 experiments
    # for i, test_image in enumerate(test_images_prob):
        N =  1024
        output = 0
        dut.reset.value = 1
        await RisingEdge(dut.clk)
        
        random_array_din = np.random.random(test_image.shape)
        random_array0 = np.random.random(fc1_weight.shape)
        random_array1 = np.random.random(fc2_weight.shape)

        binary_array_din = (random_array_din < test_image).astype(int)[::-1]
        binary_array0 = (random_array0 < fc1_weight).astype(int)[:,::-1]
        binary_array1 = (random_array1 < fc2_weight).astype(int)[:,::-1]
        
        # binary_array_din = (0 < test_image).astype(int)[::-1]
        # binary_array0 = (0 < fc1_weight).astype(int)[:,::-1]
        # binary_array1 = (0 < fc2_weight).astype(int)[:,::-1]
        
        dut.din.value = int(np.sum(binary_array_din * 2**np.arange(binary_array_din.shape[0]), axis=0))
        dut.weight_0.value = np.sum(binary_array0 * 2**np.arange(binary_array0.shape[1]), axis=1).tolist()
        dut.weight_1.value = np.sum(binary_array1 * 2**np.arange(binary_array1.shape[1]), axis=1).tolist()

        # dut.weight_0.value = [1 for x in func(layer_weights[0].T)]
        # dut.weight_1.value = [1 for x in func(layer_weights[1].T)]
        
        await RisingEdge(dut.clk)
        dut.reset.value = 0

        for _ in range(N):
            
            random_array_din = np.random.random(test_image.shape)
            random_array0 = np.random.random(fc1_weight.shape)
            random_array1 = np.random.random(fc2_weight.shape)

            binary_array_din = (random_array_din < test_image).astype(int)
            binary_array0 = (random_array0 < fc1_weight).astype(int)
            binary_array1 = (random_array1 < fc2_weight).astype(int)
            
            # binary_array_din = (0 < test_image).astype(int)[::-1]
            # binary_array0 = (0 < fc1_weight).astype(int)[:,::-1]
            # binary_array1 = (0 < fc2_weight).astype(int)[:,::-1]
           
            dut.din.value = int(np.sum(binary_array_din * 2**np.arange(binary_array_din.shape[0]), axis=0))
            dut.weight_0.value = np.sum(binary_array0 * 2**np.arange(binary_array0.shape[1]), axis=1).tolist()
            dut.weight_1.value = np.sum(binary_array1 * 2**np.arange(binary_array1.shape[1]), axis=1).tolist()

            # await FallingEdge(dut.clk)
            await RisingEdge(dut.clk)

            # print("input", dut.din.value)
            # print("weight", dut.weight_0.value[4])
            # print("mem1", dut.mem1.value[0])
            # print("count", dut.count.value)
            # print("states", dut.states.value)
            # print("layer 1", dut.layer1_output.value)
            # print("layer 2", dut.dout.value)
            
            # print("weight", dut.weight_1.value[6])
            # print("output layer", dut.layer1_output.value)

            # Calculate expected output based on select
            # if "x" in str(dut.dout.value):
                
            #     count += 1
            # else:
            #     output += dut.dout.value
            output += np.array((list(str(dut.dout.value)))).astype(int)
            # try:
            #     output += np.array((list(str(dut.dout.value)))).astype(int)
            # except:
            #     n -= 1

        pc = output / N
        result.append(pc[0])
    
        # print(f"Test {i}:")
        # print(f"Label: {y_test[a+i]} \t Expected Prob: {model(X_test[a+i:a+i+1])[0]} \t Prob Output: {pc}")
    print("N: ",N)
    print(y_test[a:b].numpy().astype(int))
    print((np.array(result) > 0.5).astype(int))
    print(np.array([y[0] for y in (model(X_test[a:b]) > 0.5)]).astype(int))
    print("accuracy1", accuracy_score(y_test[a:b].numpy().astype(int), (np.array(result) > 0.5).astype(int)))
    print("accuracy2", accuracy_score(y_test[a:b].numpy().astype(int), np.array([y[0] for y in (model(X_test[a:b]) > 0.5)])))
    print("sample", len(y_test[a:b]))


'''
2. Pytest Setup
'''

from cocotb_test.simulator import run
import pytest
import glob


def test_network():

    run(
        verilog_sources=glob.glob('hdl/*'),
        toplevel="sc_mnist_network_digit",    # top level HDL
        
        module="test_network_digit", # name of the file that contains @cocotb.test() -- this file
        simulator="icarus"

        # parameters=parameters,
        # extra_env=parameters,
        # sim_build="sim_build/sc_add/" + ",".join((f"{key}={value}" for key, value in parameters.items())),
    )