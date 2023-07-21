import numpy as np

import os
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
import onnx
import logging
import onnxruntime
from tvm.relay.transform import mixed_precision
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
import torch
from tvm.contrib import rpc

class OnnxWrapper(object):
    def __init__(self, session):
        super(OnnxWrapper, self).__init__()
        #  import onnxruntime as ort
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_names = []
        for output in session.get_outputs():
            self.output_names.append(output.name)

        try:
            seqs = [int(output_name) for output_name in self.output_names]
            sorted_seq_inds = sorted(range(len(seqs)), key=lambda k: seqs[k])
            self.output_names = [self.output_names[ind] for ind in sorted_seq_inds]
        except Exception as e:
            warnings.warn(str(e))


    def forward(self, x, **kwargs):
        outputs = self.session.run(self.output_names, {self.input_name: x})
        return outputs

import sys

logging.getLogger("auto_scheduler").setLevel(logging.DEBUG)
logging.getLogger("auto_scheduler").addHandler(logging.StreamHandler(sys.stdout))

def get_network(weight, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
    """Get the symbol definition and random weight of a network"""

    input_shape = (batch_size, 3, 320, 320)
    onnx_model = onnx.load(weight)
    input_name = "input"
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    desired_layouts = {'nn.conv2d': ['NHWC', 'default'], 'image.resize2d': ['NHWC'], 'nn.upsampling': ['NHWC']}
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(), relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    mod = tvm.IRModule.from_expr(mod["main"])
    mod = tvm.relay.transform.FastMath()(mod)
    mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
    BindPass = tvm.relay.transform.function_pass(lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(fn, params), opt_level=1)
    mod = BindPass(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)
    mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)

    mod = tvm.relay.transform.InferType()(mod)
    mod = tvm.relay.transform.ToMixedPrecision()(mod)
    mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)
    mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)

    return mod, params, input_shape


#  weight = '/home/acer/nfs-share/autoslim_repvgg_a0_gfl_dsl_v2_320_320_epoch_150_calibrate_0.5.onnx'
weight = '/home/acer/nfs-share/play_tvm/autoslim_repvgg_a0_ddod_320_320_epoch_160_calibrate_0.5.onnx'
network = os.path.basename(weight).replace('.onnx', '')
use_sparse = False
batch_size = 1
layout = "NHWC"
#  target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu cuda -arch=sm_53")
target = tvm.target.Target("cuda -arch=sm_53", host='llvm -mtriple=aarch64-linux-gnu')
#  set_cuda_target_arch('sm_53')
#  target = tvm.target.cuda(arch='sm_53')
dtype = "float16"
#  log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
log_file = "xxxx.json"

#################################################################
# Extract Search Tasks
# --------------------
# Next, we extract the search tasks and their weights from a network.
# The weight of a task is the number of appearances of the task's subgraph
# in the whole network.
# By using the weight, we can approximate the end-to-end latency of the network
# as :code:`sum(latency[t] * weight[t])`, where :code:`latency[t]` is the
# latency of a task and :code:`weight[t]` is the weight of the task.
# The task scheduler will just optimize this objective.

# Extract tasks from the network
print("Get model...")
mod, params, input_shape = get_network(
    weight,
    batch_size,
    layout,
    dtype=dtype,
    use_sparse=use_sparse,
)


print("Extract tasks...")
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    #  print(task.compute_dag)
    print(task.hardware_params)
assert(0)

#################################################################
# Begin Tuning
# ------------
# Now, we set some options for tuning and launch the search tasks
#
# * :code:`num_measure_trials` is the number of measurement trials we can use during the tuning.
#   You can set it to a small number (e.g., 200) for a fast demonstrative run.
#   In practice, we recommend setting it around :code:`800 * len(tasks)`,
#   which is typically enough for the search to converge.
#   For example, there are 29 tasks in resnet-50, so we can set it as 20000.
#   You can adjust this parameter according to your time budget.
# * In addition, we use :code:`RecordToFile` to dump measurement records into a log file,
#   The measurement records can be used to query the history best, resume the search,
#   and do more analyses later.
# * see :any:`auto_scheduler.TuningOptions`,
#   :any:`auto_scheduler.LocalRunner` for more parameters.
#



def run_tuning():
    print("Begin tuning...")
    #  measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=25600,  # change this to 20000 to achieve the best performance
        builder = auto_scheduler.LocalBuilder(timeout=1000),
        #  runner=measure_ctx.runner,
        runner = auto_scheduler.RPCRunner('nano', '10.36.172.151', 9190, timeout=1000, n_parallel=2),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

run_tuning()
assert(0)



#################################################################
# Compile and Evaluate
# --------------------
# After auto-tuning, we can compile the network with the best schedules we found.
# All measurement records are dumped into the log file during auto-tuning,
# so we can read the log file and load the best schedules.

# Compile with the history best
print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

lib.export_library('gfl_nhwc_fp16.tar')
assert(0)


# Create graph executor
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("data", data_tvm)

# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))


#################################################################
# Other Tips
# ----------
# 1. During the tuning, the auto-scheduler needs to compile many programs and
#    extract feature from them. This part is CPU-intensive,
#    so a high-performance CPU with many cores is recommended for faster search.
# 2. You can use :code:`python3 -m tvm.auto_scheduler.measure_record --mode distill -i log.json`
#    to distill the large log file and only save the best useful records.
# 3. You can resume a search from the previous log file. You just need to
#    add a new argument :code:`load_log_file` when creating the task scheduler
#    in function :code:`run_tuning`. Say,
#    :code:`tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)`
# 4. If you have multiple target CPUs, you can use all of them for measurements to
#    parallelize the measurements. Check this :ref:`section <tutorials-autotvm-scale-up-rpc-tracker>`
#    to learn how to use the RPC Tracker and RPC Server.
#    To use the RPC Tracker in auto-scheduler, replace the runner in :code:`TuningOptions`
#    with :any:`auto_scheduler.RPCRunner`.

