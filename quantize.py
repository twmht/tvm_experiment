import tvm
from tvm import te
from tvm import relay
import mxnet as mx
from tvm.contrib.download import download_testdata
from mxnet import gluon
import logging
from tvm.contrib import graph_executor
import os

batch_size = 1
model_name = "resnet18_v1"
target = "cuda"
dev = tvm.device(target)

calibration_rec = download_testdata(
    "http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/data/val_256_q90.rec",
    "val_256_q90.rec",
)


def get_val_data(num_workers=4):
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch):
        return batch.data[0].asnumpy(), batch.label[0].asnumpy()

    img_size = 299 if model_name == "inceptionv3" else 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec=calibration_rec,
        preprocess_threads=num_workers,
        shuffle=False,
        batch_size=batch_size,
        resize=256,
        data_shape=(3, img_size, img_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )
    return val_data, batch_fn

calibration_samples = 10


def calibrate_dataset():
    val_data, batch_fn = get_val_data()
    val_data.reset()
    for i, batch in enumerate(val_data):
        if i * batch_size >= calibration_samples:
            break
        data, _ = batch_fn(batch)
        yield {"data": data}

def get_model():
    gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == "inceptionv3" else 224
    data_shape = (batch_size, 3, img_size, img_size)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    return mod, params

def quantize(mod, params, data_aware):
    if data_aware:
        with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="max"):
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    else:
        with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    return mod

def run_inference(mod):
    model = relay.create_executor("vm", mod, dev, target).evaluate()
    val_data, batch_fn = get_val_data()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch)
        prediction = model(data)
        if i > 10:  # only run inference on a few samples in this tutorial
            break


def main():
    #  mod, params = get_model()
    #  mod = quantize(mod, params, data_aware=True)
    #  with tvm.transform.PassContext(opt_level=3):
        #  lib = relay.build_module.build(mod, target=target, params=params)
    #  lib.export_library('/home/acer/nfs-share/play_tvm/deploy_fp32.so')

    #  lib = tvm.runtime.load_module('/home/acer/nfs-share/play_tvm/deploy_fp32.so')
    lib = tvm.runtime.load_module('/home/acer/nfs-share/play_tvm/deploy_int8.so')
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib['default'](dev))
    print(module.benchmark(dev, number=1, repeat=600))
    #  model.run()
    #  output = model.get_output(0)
    #  input = model.get_input(0)
    #  print(input.shape)
    #  print(model.get_input_index('data'))
    #  print(model.get_num_outputs())
    #  print(model.get_num_inputs())
    #  print(output.shape)
    #  print(model.get_num_inputs())
    #  module = model.module
    #  input_names = list(module.params.keys())
    #  print(input_names)
    #  val_data, batch_fn = get_val_data()
    #  for i, batch in enumerate(val_data):
        #  data, label = batch_fn(batch)
        #  print(data)
        #  assert(0)
        #  model.set_input("input", data)
        #  model.run()
        #  output = model.get_output(0)
        #  print(output)
        #  print(prediction)
#  data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
#  module.run()
#  output = module.get_output(0)
        #  if i > 10:  # only run inference on a few samples in this tutorial
            #  break

    #  run_inference(mod)


if __name__ == "__main__":
    main()
