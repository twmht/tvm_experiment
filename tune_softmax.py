import tvm
from tvm import te, topi
import tvm.testing
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple

_softmax_schedule = {
    "generic": topi.generic.schedule_softmax,
    "cpu": topi.x86.schedule_softmax,
    "gpu": topi.cuda.schedule_softmax,
    "hls": topi.hls.schedule_softmax,
}

shape = (1,10)
dtype = 'float32'
A = te.placeholder(shape, dtype=dtype, name='A')
B = topi.nn.softmax(A, axis=1)
target = tvm.target.cuda(arch='sm_61')
#  target = tvm.target.Target('llvm')
with tvm.target.Target(target):
    fschedule = tvm.topi.testing.dispatch(target, _softmax_schedule)
    s = fschedule(B)

ir_m = tvm.lower(s, [A], simple_mode=True)
rt_m = tvm.build(ir_m, [A], 'cuda')
#  print(ir_m)
#  print(ir_m)
print(rt_m.imported_modules[0].get_source())
#  with tvm.transform.PassContext(opt_level=3):
    #  f = tvm.build(s, [A, B], target)

#  print(f.get_source())

#  dev = tvm.device(str(target), 0)
#  a_np = np.random((1,16,256,256)).astype(np.float32)
#  a = tvm.nd.array(a_np, dev)
#  b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)

#  f(a, b)
