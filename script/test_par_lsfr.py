import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge

@cocotb.test()
async def print_randoms_test(dut):
    # Start the clock
    clock = Clock(dut.clk, 100, units="us")  # 10 us period for the clock
    cocotb.start_soon(clock.start())

    # Reset the device
    dut.reset.value = 1
    await FallingEdge(dut.clk)
    dut.reset.value = 0

    # Wait for a few clock cycles and print the random numbers
    for _ in range(10):  # Adjust the range for more or fewer cycles
        await RisingEdge(dut.clk)
        # Print the random numbers after each clock cycle
        print(f"Random: {dut.random.value}")
        # print(dut.intermediate.value)
        # print(dut.feedback.value)

import glob, os
from cocotb.runner import get_runner

def test_():

    verilog_sources = [
        "src/components/parallel_lfsr_8bit_8x.sv", 
        ]
    
    sim = os.getenv("SIM", "icarus")

    runner = get_runner(sim)
    
    # runner.flags = ["-sv"]
    
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="parallel_lfsr_8bit_8x",
        always=True,
    )
    runner.test(
        hdl_toplevel="parallel_lfsr_8bit_8x", 
        test_module="test_par_lsfr", # this file name,
    )

if __name__ == "__main__":
    test_()