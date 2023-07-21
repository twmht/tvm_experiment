import tvm
#  import torch
import cupy as cp

example_input_tensor = cp.zeros((1,3,112,112), dtype=cp.float32)

input_tvm = tvm.nd.from_dlpack(cp.ndarray.toDlpack(example_input_tensor))
print(input_tvm)
print(input_tvm.device)
