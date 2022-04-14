import pickle

# Load
with open('artifacts/cnn_speech_commands/example_cnn_speech_commands_struct_pru_agingevosearch_state.pickle', 'rb') as f:
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

output_dir = "tmp/tflite"

input_shape = (49, 40, 1)
num_classes = 10



def convert_to_tflite(arch, output_file):
    model = arch.to_keras_model(input_shape, num_classes)
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()
 
    with open(output_file, "wb") as f:
        f.write(tflite_model)

#convert
archid = 1327
arch = EvaluatedPoint[archid].point.arch
convert_to_tflite(arch, output_file=f"{output_dir}/speech_command_EvaluatedPoint[{archid}]_point_arch_nq.tflite")
