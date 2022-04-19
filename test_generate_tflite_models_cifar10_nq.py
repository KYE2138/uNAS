import pickle

# Load pickle
with open('storage/eliberis/uNAS/artifacts/cnn_cifar10/example_cnn_cifar10_struct_pru_agingevosearch_state.pickle', 'rb') as f:
  EvaluatedPoint = pickle.load(f)
print("------------------------------")
print("len of EvaluatedPoint(search models):",len(EvaluatedPoint))
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

import os, sys
import time

# get time 
timestr = time.strftime("%Y%m%d-%H%M%S")
# make new dir to save search(converted) model
dataset_name= "cifar10"
output_dir = f"tmp/tflite/{dataset_name}/{timestr}"
os.makedirs(output_dir)
print (f"output dir:{output_dir}")

# set parameter to convert model
input_shape = (49, 40, 1)
num_classes = 10
model_format = "nq"


# convert function
def convert_to_tflite(arch, output_file):
    model = arch.to_keras_model(input_shape, num_classes)
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converted_model = converter.convert()
    with open(output_file, "wb") as f:
        f.write(converted_model)

# generate_model
def generate_model(archid=-1):
    # if archid is not set, convert all search models in EvaluatedPoint
    if archid = -1:
        for archid in range(len(EvaluatedPoint)):
            arch = EvaluatedPoint[archid].point.arch
            convert_to_tflite(arch, output_file=f"{output_dir}/{dataset_name}-{archid}-{model_format}.tflite")
    elif:
        convert_to_tflite(arch, output_file=f"{output_dir}/{dataset_name}-{archid}-{model_format}.tflite")

generate_model(0)

