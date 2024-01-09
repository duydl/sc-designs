import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
from cocotb.result import TestFailure

@cocotb.test()
async def test_lfsr(dut):
    clock = Clock(dut.clk, 10, units="ns")  # Create a 10ns clock
    cocotb.start_soon(clock.start(start_high=False))  # Start the clock

    dut.rst.value = 1  # Assert reset
    await FallingEdge(dut.clk)
    dut.rst.value = 0  # Deassert reset

    # Wait for a few clock cycles to observe the LFSR output
    for _ in range(20):
        await RisingEdge(dut.clk)

    # Perform checks on LFSR output
    for _ in range(100):
        await RisingEdge(dut.clk)
        lfsr_value = int(dut.lfsr_out.value)
        print(lfsr_value)
        # Perform checks on LFSR output to verify the pseudo-randomness
        # if lfsr_value == 0 or lfsr_value == 255:
        #     raise TestFailure(f"LFSR produced invalid output: {lfsr_value}")


from cocotb_test.simulator import run
import glob

def test_():

    run(
        verilog_sources=glob.glob('hdl/*'),
        toplevel="lfsr",    # top level HDL
        module="test_lfsr", # name of the file that contains @cocotb.test() -- this file
        simulator="icarus"
    )



# import glob, os
# from cocotb.runner import get_runner

# def test_():

#     verilog_sources = glob.glob('hdl/*') 
#     sim = os.getenv("SIM", "icarus")

#     runner = get_runner(sim)
#     runner.build(
#         verilog_sources=verilog_sources,
#         hdl_toplevel="lfsr",
#         always=True,
#     )
#     runner.test(
#         hdl_toplevel="lfsr", test_module="test_lfsr"
#     )


if __name__ == "__main__":
    test_()