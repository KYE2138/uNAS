import pickle

# Load
with open('artifacts/cnn_speech_commands/example_cnn_speech_commands_struct_pru_agingevosearch_state.pickle', 'rb') as f:
  end_point = pickle.load(f)
print("------------------------------")
print("len of end_point:",len(end_point))
print("------------------------------")
  
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

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = \
        lambda: [[np.random.random((1,) + input_shape).astype("float32")] for _ in range(5)]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()

 
    with open(output_file, "wb") as f:
        f.write(model_bytes)
for i in range(1, 21):
  cnn_arch = end_point[i*100-1].point.arch
  convert_to_tflite(cnn_arch, output_file=f"{output_dir}/speech_command_end_point[{i*100-1}]_point_arch.tflite")
