import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles

import numpy as np
import matplotlib.pyplot as plt

'''
1. Testbench
'''

@cocotb.coroutine
def apply_pulse_signal(dut, pulse_signal):
    bitstream = []
    for i in range(len(pulse_signal)):
        dut.pulse_input.value = int(pulse_signal[i])
        yield RisingEdge(dut.clk)
        bit_ = int(dut.bit_stream.value)
        bitstream.append(bit_)
    return bitstream

@cocotb.test()
async def sc_pulse_tb(dut):
    
    # Time parameters
    duration = 0.0255  # sec
    sampling_rate = 1000000  # 1000 kHz i.e clock cycle 1us
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    print("Sample Num: ", len(t))

    clock = Clock(dut.clk, 1/sampling_rate, units="sec")  
    
    # Start the clock. Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(clock.start(start_high=False))
    
    await RisingEdge(dut.clk)
    
    # Pulse train parameters
    amplitudes = np.array([1, 0.1, 1/255]) # 255, 25.5, 1
    pulse_frequencies = amplitudes  * 10000 # 10kHz for 255 pixel i.e period 100 us
    pulse_width = 1 / 100000  # 10us

    # Create multiple pulse train signals with constant pulse width in the time domain
    for i, (pulse_frequency, amplitude) in enumerate(zip(pulse_frequencies, amplitudes)):
        pulse_period = 1 / pulse_frequency
        pulse_signal = 1 * np.where((t % pulse_period) < (pulse_width), 1, 0)
        bitstream = await apply_pulse_signal(dut, pulse_signal)
        # for i in range(len(pulse_signal)):
        #     dut.pulse_input.value = int(pulse_signal[i])
        #     await RisingEdge(dut.clk)
        
        print(bitstream[0:200])
        print(sum(pulse_signal))
        print(sum(bitstream))


'''
2. Pytest Setup
'''

from cocotb_test.simulator import run
import glob


def test_pulse():

    run(
        verilog_sources=glob.glob('hdl/*'),
        toplevel="sc_pulse", # top level HDL
        module="test_pulse", # name of file contains @cocotb.test
        simulator="icarus"
    )