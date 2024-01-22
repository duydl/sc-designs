import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles

import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Neural network model
class LunarLanderNetwork(nn.Module):
    def __init__(self):
        super(LunarLanderNetwork, self).__init__()

        self.num_states = 8
        self.hidden_units = 256
        self.num_actions = 4
        
        # The hidden layer
        self.fc1 = nn.Linear(in_features = self.num_states, out_features = self.hidden_units)
        
        # The output layer
        self.fc2 = nn.Linear(in_features = self.hidden_units, out_features = self.num_actions)
        
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        # No activation func, output should be a tensor(batch, num_actions)
        out = self.fc2(x)
        
        return out

'''
1. Testbench
'''

@cocotb.test()
async def sc_drl(dut):
    dut.threshold.value = 10
    clock = Clock(dut.clk, 10, units="us")  # Create a 10us period clock on port clk
    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))

    model_path = '/home/ubuntu20_1/Projects_Ubuntu20/sc_designs/script/lunar_landing_model_700.pth'
    model = LunarLanderNetwork()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Create the LunarLander environment
    env = gym.make('LunarLander-v2', render_mode="human")

    # Number of episodes to run
    num_episodes = 5
    await RisingEdge(dut.clk)
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        await RisingEdge(dut.clk)
        while not done:
            await RisingEdge(dut.clk)
            state_tensor = torch.from_numpy(state).float()
            action_probs = model(state_tensor)
            action = torch.argmax(action_probs).item()
            
            print(env.step(action))
            next_state, reward, done, trunc, info = env.step(action)

            state = next_state

    print(f"Episode {episode + 1} finished")

    env.close()  # Close the environment



import glob, os
from cocotb.runner import get_runner

def test_abs():

    verilog_sources = glob.glob('hdl/*') 
    # proj_path = Path(__file__).resolve().parent.parent
    
    # verilog_sources = [proj_path / "hdl" / "sc_abs.sv"]
    
    sim = os.getenv("SIM", "icarus")

    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="lfsr_rng",
        always=True,
    )
    runner.test(
        hdl_toplevel="lfsr_rng", test_module="test__rl"
    )

if __name__ == "__main__":
    sc_drl()
# '''
# 2. Pytest Setup
# '''

# from cocotb_test.simulator import run
# import glob


# def test_network():

#     run(
#         verilog_sources=glob.glob('hdl/*'),
#         toplevel="lfsr_rng",    # top level HDL
        
#         module="test__rl", # name of the file that contains @cocotb.test() -- this file
#         simulator="icarus"

#         # parameters=parameters,
#         # extra_env=parameters,
#         # sim_build="sim_build/sc_add/" + ",".join((f"{key}={value}" for key, value in parameters.items())),
#     )