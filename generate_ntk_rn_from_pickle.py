
import pdb
import os
import argparse
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf

from architecture import Architecture
from cnn import CnnSearchSpace
from resource_models.models import model_size, peak_memory_usage

import csv
from model_trainer import ModelTrainer
from metrics_file_ntk_rn import ModelMetricsFile


def main():
    # 取得參數
    parser = argparse.ArgumentParser("uNAS Search")
    #configs/cnn_cifar10_struct_pru.py
    parser.add_argument("config_file", type=str, help="A config file describing the search parameters")
    parser.add_argument("--name", type=str, help="Experiment name (for disambiguation during state saving)")
    parser.add_argument("--load_from", type=str, default=None, help="A search state file to resume from")
    parser.add_argument("--save_to", type=str, default=None, help="A search state file to save to")
    parser.add_argument("--save_every", type=int, default=5, help="After how many search steps to save the state")
    parser.add_argument("--seed", type=int, default=0, help="A seed for the global NumPy and TensorFlow random state")
    parser.add_argument("--metric_type", type=str, nargs='+', default=["ntk","rn"], help="Some metrics list by the pickle")
    parser.add_argument("--input_shape", type=int, nargs='+', default=[32,32,3], help="A input shape of the model")
    parser.add_argument("--num_classes", type=int, default=10, help="A num classes of the model")
    parser.add_argument("--range_points", type=int, nargs='+', default=[0,-1], help="A range of points")

    
    args = parser.parse_args()
    # 執行config_file(.py)內之code,configs 則是全域變數(以字典型態儲存)
    configs = {}
    exec(Path(args.config_file).read_text(), configs)
    #configs.training_config
    #configs.training_config.pruning
    #configs.search_config
    #configs.bound_config
    training_config = configs["training_config"]
    trainer = ModelTrainer(training_config)

    # Load pickle
    pickle_load_from_path = args.load_from
    with open(pickle_load_from_path, 'rb') as f:
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
    
    #del metrics
    save_path = './tmp/metrics/ntk_rn'
    train_loader_save_path = f'{save_path}/train_loader.pickle'
    val_loader_save_path = f'{save_path}/val_loader.pickle'
    if os.path.isfile(train_loader_save_path) and os.path.isfile(val_loader_save_path):
        print (f"train_loader_save_path is already exist:{train_loader_save_path}")
        print (f"val_loader_save_path is already exist:{val_loader_save_path}")
        os.remove(train_loader_save_path)
        os.remove(val_loader_save_path)

    
    '''
    input_shape = args.input_shape
    num_classes = args.num_classes

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
    '''

    
    
    pickle_save_to_path = args.save_to
    metric_type = args.metric_type
    input_shape = args.input_shape
    num_classes = args.num_classes
    range_points = args.range_points
    range_points = list(map(int, range_points))
    #pdb.set_trace()
    # 開啟
    with open(pickle_save_to_path, "w", newline="") as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(["id", "val_acc", "test_acc", "peak_memory_usage", "model_size", "inference_latency", "ntk", "rn"])
        EvaluatedPoint_num = len(EvaluatedPoint)
        if range_points[1] == -1:
            range_points[1] = EvaluatedPoint_num
        for i in range(range_points[0], range_points[1]):
            val_error = EvaluatedPoint[i].val_error
            test_error = EvaluatedPoint[i].test_error
            resource_features = EvaluatedPoint[i].resource_features
            arch = EvaluatedPoint[i].point.arch

            # ntks & rns
            model = arch.to_keras_model(input_shape, num_classes)
            model_rn = arch.to_keras_model((2, 2, 1), num_classes)
            ntks, rns= ModelMetricsFile(trainer).get_metrics(model=model, model_rn=model_rn, num_batch=1, num_networks=3)

            if "ntk" in metric_type:
                ntk = np.mean(ntks).astype('int64')
                resource_features.append(ntk)

            if "rn" in metric_type:
                rn = np.mean(rns).astype('int64')
                # max rn ~ 3000
                # 限制rn在1500以上
                rn = 4000-rn
                resource_features.append(rn)


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

if __name__ == "__main__":
    main()