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
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# Define a simple feedforward neural network with one hidden layer and tanh activation
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # One hidden layer with 64 units
        self.fc2 = nn.Linear(128, 1)  # Output layer with 2 classes
        self.tanh = nn.Tanh() 
        self.sigmoid = nn.Sigmoid()# Sigmoid activation for binary classification
        
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.requires_grad = False
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.tanh(self.fc1(x))  # Apply tanh activation
        x = self.fc2(x)
        x = (self.tanh(x)+1)/2
        return x

# Define data transformations and load the MNIST test dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Define a function to filter the test dataset to include only two classes (e.g., 3 and 7)
def filter_dataset(dataset, class1, class2):
    filtered_data = []
    for data, label in dataset:
        if label == class1 or label == class2:
            filtered_data.append((data, label))
    return filtered_data

# Filter the test dataset to include only classes 3 and 7
class1 = 0
class2 = 1


model = Net()
criterion = nn.BCELoss() 
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


filtered_trainset = filter_dataset(trainset, class1, class2)

# Create a DataLoader for the filtered dataset
trainloader = torch.utils.data.DataLoader(filtered_trainset, batch_size=64, shuffle=True)
# Training loop
for epoch in range(10):  # You can adjust the number of epochs
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        labels = labels.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

print("Binary training with one hidden layer and tanh activation finished")

filtered_testset = filter_dataset(testset, class1, class2)

# Create a DataLoader for the filtered test dataset
testloader = torch.utils.data.DataLoader(filtered_testset, batch_size=64, shuffle=False)
# Evaluation loop
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        
        outputs = model(images)
        predicted = (outputs >= 0.5).float()  # Apply threshold of 0.5
        # predicted = torch.cat(predicted, dim=0)
        # print(labels.view(-1, 1))
        # print(predicted)
        # print("labels.size(0)", labels.size(0))
        # print("(predicted == labels).sum().item()", (predicted == labels).sum().item())
        total += labels.size(0)
        correct += (predicted == labels.view(-1, 1)).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on the test dataset: {accuracy:.2f}%')



image_data = []
labels = []
for data in testloader:
    images, batch_labels = data
    image_data.append(images)  # Convert tensor to numpy array
    labels.append(batch_labels.numpy())  # Convert tensor to numpy array

# Concatenate the lists of arrays to get a single array for image data and labels
image_data = torch.cat(image_data, dim=0)
X_test = np.concatenate(image_data.numpy())
y_test = np.concatenate(labels)

model_state_dict = model.state_dict()
# print((model_state_dict['fc1.weight'].clone() + 1) / 2)
# print((model_state_dict['fc2.weight'].clone() + 1) / 2)

# fc1_bias = (model_state_dict['fc1.bias'].numpy()+ 1) / 2
# fc2_bias = (model_state_dict['fc2.bias'].numpy()+ 1) / 2
fc1_weight = (model_state_dict['fc1.weight'].numpy() + 1) / 2
fc2_weight = (model_state_dict['fc2.weight'].numpy() + 1) / 2
test_images_prob = (1 + X_test)/2


'''
1. Testbench
'''

@cocotb.test()
async def sc_network_tb(dut):
 

    
    clock = Clock(dut.clk, 10)  

    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))


    a, b = 0,20
    result = []
    
    for i, test_image in enumerate(test_images_prob[a:b]):
        N = 256
        output = 0
        dut.reset.value = 1
        await RisingEdge(dut.clk)
        dut._log.info("Running test!")
        test_image = test_image.reshape(28*28)
        random_array_din = np.random.random(test_image.shape)
        random_array0 = np.random.random(fc1_weight.shape)
        random_array1 = np.random.random(fc2_weight.shape)

        binary_array_din = (random_array_din < test_image).astype(int)[::-1]
        binary_array0 = (random_array0 < fc1_weight).astype(int)[:,::-1]
        binary_array1 = (random_array1 < fc2_weight).astype(int)[:,::-1]
        
        dut.din.value = int(np.sum(binary_array_din * 2**np.arange(binary_array_din.shape[0]), axis=0))
        dut.weight_0.value = np.sum(binary_array0 * 2**np.arange(binary_array0.shape[1]), axis=1).tolist()
        dut.weight_1.value = np.sum(binary_array1 * 2**np.arange(binary_array1.shape[1]), axis=1).tolist()
        dut._log.info("Running test!")
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
    
        print(f"Test {i}:")
        print(f"Label: {y_test[a+i]} \t Expected Prob: {model(image_data[a+i:a+i+1])[0]} \t Prob Output: {pc}")

    print(y_test[a:b].numpy().astype(int))
    print((np.array(result) > 0.5).astype(int))
    print(np.array([y[0] for y in (model(image_data[a:b]) > 0.5)]).astype(int))
    print("accuracy1", accuracy_score(y_test[a:b].numpy().astype(int), (np.array(result) > 0.5).astype(int)))
    print("accuracy2", accuracy_score(y_test[a:b].numpy().astype(int), np.array([y[0] for y in (model(image_data[a:b]) > 0.5)])))
    print("sample", len(y_test[a:b]))


'''
2. Pytest Setup
'''

from cocotb_test.simulator import run
import glob


def test_network():

    run(
        verilog_sources=glob.glob('hdl/*'),
        toplevel="sc_mnist_network",    # top level HDL
        
        module="test_network", # name of the file that contains @cocotb.test() -- this file
        simulator="icarus"

        # parameters=parameters,
        # extra_env=parameters,
        # sim_build="sim_build/sc_add/" + ",".join((f"{key}={value}" for key, value in parameters.items())),
    )
    
    
    
