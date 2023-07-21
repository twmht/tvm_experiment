import numpy as np

import tvm
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
from tvm.contrib import graph_executor
import onnx
import logging
import onnxruntime

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

logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
    """Get the symbol definition and random weight of a network"""

    input_shape = (batch_size, 3, 320, 320)
    #  onnx_model = onnx.load('/home/acer/nfs-share/play_tvm/autoslim_repvgg_a0_ddod_random_320_320_ratio_0.5_random.onnx')
    #  onnx_model = onnx.load('/home/acer/nfs-share/play_tvm/autoslim_repvgg_a0_ddod_320_320_epoch_160_calibrate_0.5.onnx')
    #  onnx_model = onnx.load('/home/acer/vargfacenet_webface12m_112_112_epoch_20.onnx')
    onnx_model = onnx.load('/home/acer/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_v2_deploy_act_320_320_best_coco_bbox_mAP_epoch_291.onnx')
    input_name = "input"
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return mod, params, input_shape


network = "vargfacenet_nchw"
use_sparse = False
batch_size = 1
layout = "NCHW"
target = tvm.target.cuda('cuda -arch=sm_61')
dtype = "float32"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)

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
    network,
    batch_size,
    layout,
    dtype=dtype,
    use_sparse=use_sparse,
)

#  desired_layouts = {'nn.conv2d': ['NHWC', 'default']}

#  seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(), relay.transform.ConvertLayout(desired_layouts)])

#  with tvm.transform.PassContext(opt_level=3):
    #  mod = seq(mod)
    #  lib = relay.build(mod, target=target, params=params)

#  lib.export_library('/home/acer/nfs-share/play_tvm/deploy_fp32.so')
#  lib = tvm.runtime.load_module('/home/acer/nfs-share/play_tvm/deploy_fp32.so')
#  dev = tvm.device(str(target), 0)
#  module = graph_executor.GraphModule(lib['default'](dev))

#  data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
#  module.set_input("input", data_tvm)
#  module.run()
#  output = module.get_output(0)

#  print(output.shape)
#  print(type(output))
#  print(output.device)
#  assert(0)


print("Extract tasks...")
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

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
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=25600,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

run_tuning()

assert(0)

######################################################################
# .. note:: Explain the printed information during tuning
#
#   During the tuning, a lot of information will be printed on the console.
#   They are used for debugging purposes. The most important info is the output
#   of the task scheduler. The following table is a sample output.
#
#   .. code-block:: c
#
#     ----------------------------------------------------------------------
#     ------------------------------  [ Task Scheduler ]
#     ----------------------------------------------------------------------
#     |  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
#     -------------------------------------------------
#     |    0 |        0.010 |           0.40 |     64 |
#     |    1 |        0.087 |          47.19 |     64 |
#     |    2 |        0.008 |          -0.00 |     64 |
#     |    3 |        0.177 |         582.07 |     64 |
#     |    4 |        0.268 |         862.37 |    256 |
#     |    5 |        0.166 |         621.13 |    128 |
#     |    6 |        0.170 |         605.10 |    128 |
#     |    7 |        0.128 |         403.20 |     64 |
#     |    8 |        0.189 |         545.71 |     64 |
#     |    9 |        0.231 |        1001.01 |    448 |
#     |   10 |        0.155 |         664.80 |    256 |
#     |   11 |        0.155 |         662.86 |    256 |
#     |   12 |        0.119 |         434.08 |     64 |
#     |   13 |        0.199 |         522.13 |     64 |
#     |   14 |        0.235 |         986.56 |    320 |
#     |   15 |        0.149 |         689.13 |    128 |
#     |   16 |        0.155 |         664.80 |    192 |
#     |   17 |        0.151 |         340.64 |     64 |
#     |   18 |        0.176 |         597.55 |    128 |
#     |   19 |        0.220 |        1054.37 |    192 |
#     |   20 |        0.150 |         686.01 |    128 |
#     |   21 |        0.159 |         650.88 |    128 |
#     |   22 |        0.073 |         358.19 |     64 |
#     |   23 |        0.031 |          70.63 |     64 |
#     |   24 |        0.251 |         947.73 |    128 |
#     |   25 |        0.157 |         652.47 |    128 |
#     |   26 |        0.215 |         954.84 |    128 |
#     |   27 |        0.237 |         868.92 |    128 |
#     |   28 |        0.266 |         774.06 |    128 |
#     -------------------------------------------------
#     Estimated total latency: 10.016 ms      Trials: 3992    Used time : 1131 s      Next ID: 15
#
#   This table lists the latency and (estimated) speed of all tasks.
#   It also lists the allocation of measurement trials for all tasks.
#   The last line prints the total weighted latency of these tasks,
#   which can be a rough estimation of the end-to-end execution time
#   of the network.
#   The last line also prints the total number of measurement trials,
#   total time spent on auto-tuning and the id of the next task to tune.
#
#   There will also be some "tvm::Error"s errors, because the
#   auto-scheduler will try some invalid schedules.
#   You can safely ignore them if the tuning can continue, because these
#   errors are isolated from the main process.
#

######################################################################
# .. note:: Terminate the tuning earlier
#
#   You can terminate the tuning earlier by forcibly killing this process.
#   As long as you get at least one valid schedule for each task in the log file,
#   you should be able to do the compilation (the secion below).
#


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

