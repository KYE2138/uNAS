import os, sys
import time
import pickle
import csv
import numpy as np
import tensorflow as tf
from architecture import Architecture
from cnn import CnnSearchSpace
from resource_models.models import model_size, peak_memory_usage


# Load pickle
pickle_filepath ="/storage/KYE2138/uNAS/artifacts/cnn_cifar10/pre_ntk_cnn_cifar10_struct_pru_agingevosearch_state.pickle"
with open(pickle_filepath, 'rb') as f:
    EvaluatedPoint = pickle.load(f)
print("------------------------------")
print(f"len of EvaluatedPoint({pickle_filepath}):",len(EvaluatedPoint))
print("-----------------------------")
'''
@dataclass
class EvaluatedPoint:
    point: ArchitecturePoint
    val_error: float
    test_error: float
    resource_features: List[Union[int, float]]
''' 

# get time 
timestr = time.strftime("%Y%m%d_%H%M%S")
# make new dir to save search(converted) model
dataset_name= "cifar10"
output_dir = f"tmp/tflite/{dataset_name}/{timestr}"
os.makedirs(output_dir)
print (f"output dir:{output_dir}")

# set parameter to convert model
input_shape = (32, 32, 3)
num_classes = 10
model_format = "pru_ae_q_pre_ntk"

# convert function
def convert_to_tflite(arch, output_file):
    model = arch.to_keras_model(input_shape, num_classes)
    #model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = \
        lambda: [[np.random.random((1,) + input_shape).astype("float32")] for _ in range(5)]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    #convert
    converted_model = converter.convert()
    with open(output_file, "wb") as f:
        f.write(converted_model)

# generate_model
def generate_model(archid=-1):
    # if archid is not set, convert all search models in EvaluatedPoint
    if archid == -1:
        for archid in range(len(EvaluatedPoint)):
            arch = EvaluatedPoint[archid].point.arch
            convert_to_tflite(arch, output_file=f"{output_dir}/{dataset_name}_{archid}_{model_format}.tflite")
    else:
        arch = EvaluatedPoint[archid].point.arch
        convert_to_tflite(arch, output_file=f"{output_dir}/{dataset_name}_{archid}_{model_format}.tflite")

# run func
generate_model()

