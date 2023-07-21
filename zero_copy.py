import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor
from tvm import testing


dev = tvm.cuda(0)
target = tvm.target.Target("cuda")

# A simple relay func:
# 1. y = x + 1
# 2. return [x, y]
x = relay.var("x", shape=(2, 2), dtype="float32")
y = relay.add(x, relay.ones((2, 2), dtype="float32"))
func = relay.Function([x], relay.expr.Tuple([x, y]))

# Build 2 exactly same relay module.
def build_relay_module():

    mod = tvm.IRModule()
    mod["main"] = func
    lib = relay.build(mod, target=target)

    m = graph_executor.GraphModule(lib["default"](dev))

    return m

mod = build_relay_module()
mod_zero_copy = build_relay_module()

# Run these 2 modules.

# 2 same inputs.
input_nd = tvm.nd.array(np.ones((2, 2), dtype="float32"), device=dev)
input_nd_zero_copy = tvm.nd.array(np.ones((2, 2), dtype="float32"), device=dev)

# set_input() vs. set_input_zero_copy()
mod.set_input("x", input_nd)

index = mod_zero_copy.get_input_index("x")
mod_zero_copy.module["set_input_zero_copy"](index, input_nd_zero_copy)

# Run
mod.run()
mod_zero_copy.run()

# We expect 2 mod have the exactly same output "x", however...
testing.assert_allclose(mod.get_output(0).numpy(), mod_zero_copy.get_output(0).numpy())
