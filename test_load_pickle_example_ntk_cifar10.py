
import os
import argparse

# 取得參數
parser = argparse.ArgumentParser("uNAS Search")
#configs/cnn_cifar10_struct_pru.py
parser.add_argument("config_file", type=str, help="A config file describing the search parameters")
args = parser.parse_args()
# 執行config_file(.py)內之code,configs 則是全域變數(以字典型態儲存)
configs = {}
exec(Path(args.config_file).read_text(), configs)
training_config = configs["training_config"]
#load dataset
dataset = training_config.dataset

import pickle

# Load
with open('artifacts/cnn_cifar10/example_cnn_cifar10_struct_pru_agingevosearch_state.pickle', 'rb') as f:
  EvaluatedPoint = pickle.load(f)
print("------------------------------")
print("len of EvaluatedPoint:",len(EvaluatedPoint))
print("-----------------------------")
'''
@dataclass
class EvaluatedPoint:
    point: ArchitecturePoint
    val_error: float
    test_error: float
    resource_features: List[Union[int, float]]
''' 
  
import numpy as np
import tensorflow as tf

from architecture import Architecture
from cnn import CnnSearchSpace
from resource_models.models import model_size, peak_memory_usage

output_dir = "artifacts/cnn_cifar10"

input_shape = (32, 32, 3)
num_classes = 10



def get_resource_requirements(arch: Architecture):
    rg = arch.to_resource_graph(input_shape, num_classes)
    return model_size(rg), peak_memory_usage(rg, exclude_inputs=False)

def convert_to_tflite(arch, output_file):
    model = arch.to_keras_model(input_shape, num_classes)
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = \
        lambda: [[np.random.random((1,) + input_shape).astype("float32")] for _ in range(5)]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
 
    with open(output_file, "wb") as f:
        f.write(model_bytes)

import csv
#ntk
from metrics_file import ModelMetricsFile

# 開啟
with open(f"{output_dir}/example_cnn_cifar10_struct_pru_agingevosearch_state_ntk.csv", "w", newline="") as csvfile:
  wr = csv.writer(csvfile)
  wr.writerow(["id", "val_acc", "test_acc", "peak_memory_usage", "model_size", "inference_latency","ntk"])
  EvaluatedPoint_num = len(EvaluatedPoint)
  for i in range(0, EvaluatedPoint_num):
    val_error = EvaluatedPoint[i].val_error
    test_error = EvaluatedPoint[i].test_error
    resource_features = EvaluatedPoint[i].resource_features

    # pre ntk
    arch = EvaluatedPoint[i].point.arch
    model = arch.to_keras_model(input_shape, num_classes)
    ntks = ModelMetricsFile(training_config).get_metrics(model, num_batch=1)
    ntk = np.mean(ntks).astype('int64')
    resource_features.append(ntk)
    '''
    cnn_arch = EvaluatedPoint[i*100-1].point.arch
    print("------------------------------")
    print(f"val_error of speech_command_EvaluatedPoint[{i*100-1}]_point_arch:", val_error)
    print(f"test_error of speech_command_EvaluatedPoint[{i*100-1}]_point_arch:", test_error)
    print("resource_features: [peak_memory_usage, model_size, inference_latency]")
    print(f"resource_features of speech_command_EvaluatedPoint[{i*100-1}]_point_arch:", resource_features)
    print("------------------------------")
    convert_to_tflite(cnn_arch, output_file=f"{output_dir}/speech_command_EvaluatedPoint[{i*100-1}]_point_arch.tflite")
    '''

    #################### TEGNAS testntk #################### 



    EvaluatedPoint_list = [i, 1-val_error, 1-test_error]
    EvaluatedPoint_list.extend(resource_features)
    wr.writerow(EvaluatedPoint_list)

