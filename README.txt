python -m pytest -rP script/test.py

3 changes
- test file name and test module name 
- toplevel
- hdl file name and hdl module name


Wire variables need to be set with an assign outside of a procedural block. Reg variables can be set inside a procedural block. If you use SystemVerilog you can declare the variable to be of type logic and then you can assign to that either way.

ValueError: Unresolvable bit in binary string: 'x'

### Latest Error:

ValueError: Unable to accurately represent 2(us) with the simulator precision of 1e0