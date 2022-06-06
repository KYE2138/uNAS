#################### uNAS model_trainer ####################
import logging
from typing import Optional

import tensorflow as tf

from config import TrainingConfig
from pruning import DPFPruning
from utils import debug_mode

#################### save_dataset save_model wait_ntk#################### 
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import pickle
import time
import os
import pdb
import gc

#################### GPU ####################
# GPU mem issue
#config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
#sess = tf.compat.v1.Session(config=config)


class ModelMetricsFile:
    """Keras models according to the specified config."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.save_path = './tmp/metrics'

    def get_metrics(self, model, num_batch):
        dataset = self.trainer.dataset
        input_shape = self.trainer.dataset.input_shape
        num_classes = self.trainer.dataset.num_classes
        batch_size = self.trainer.config.batch_size
        model = model
        #networks_num = networks_num
        num_batch = num_batch
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print (f"save_path dir:{save_path}")
        else:
            print (f"save_path is already exist:{save_path}")


        def save_dataset(dataset, batch_size, input_shape, num_classes, num_batch, save_path):
            #################### dataset ####################
            #check loader
            train_loader_save_path = f'{save_path}/train_loader.pickle'
            val_loader_save_path = f'{save_path}/val_loader.pickle'
            if os.path.isfile(train_loader_save_path) and os.path.isfile(val_loader_save_path):
                print (f"train_loader_save_path is already exist:{train_loader_save_path}")
                print (f"val_loader_save_path is already exist:{val_loader_save_path}")
            else:
                print (f"generate train_loader_save_path :{train_loader_save_path}")
                print (f"generate val_loader_save_path :{val_loader_save_path}")
            '''   
            loader_save_path = f'{save_path}/loader.npz'
            if os.path.isfile(loader_save_path):
                print (f"loader is already exist:{loader_save_path}")
            else:
                print (f"generate loader :{loader_save_path}")
            '''                 
                # from uNAS dataset by tf
                train = dataset.train_dataset() \
                    .shuffle(100000) \
                    .batch(batch_size) \
                    .prefetch(tf.data.experimental.AUTOTUNE)
                # <PrefetchDataset shapes: ((None, 32, 32, 3), (None, 1)), types: (tf.float64, tf.uint8)>
                val = dataset.validation_dataset() \
                    .batch(batch_size) \
                    .prefetch(tf.data.experimental.AUTOTUNE)
                # <PrefetchDataset shapes: ((None, 32, 32, 3), (None, 1)), types: (tf.float64, tf.uint8)>

                # for get ntk loader input
                train_loader = []
                val_loader = []
                for i in range(num_batch):
                    # list(train.as_numpy_iterator())[0][0].shape = (128, 32, 32, 3)
                    train_input = list(train.as_numpy_iterator())[0][0]
                    # list(train.as_numpy_iterator())[0][1].shape = (128, 1)?
                    train_target = list(train.as_numpy_iterator())[0][1]
                    # targets.shape = torch.Size([128])
                    train_target = train_target.reshape((-1,))
                    # one_hot is only applicable to index tensor
                    train_target = train_target.astype(np.int64)
                    train_loader.append((train_input,train_target))

                    # for get ntk val_loader input
                    # list(val.as_numpy_iterator())[0][0].shape = (128, 32, 32, 3)
                    val_input = list(val.as_numpy_iterator())[0][0]
                    # list(val.as_numpy_iterator())[0][1].shape = (128, 1)?
                    val_target = list(val.as_numpy_iterator())[0][1]
                    # targets.shape = torch.Size([128])
                    val_target = val_target.reshape((-1,))
                    # one_hot is only applicable to index tensor
                    val_target = val_target.astype(np.int64)
                    val_loader.append((val_input,val_target))
                
                # save loader as loader.npz
                #loader_save_path = f'{save_path}/{batch_size}_{input_shape}_{num_classes}_{num_batch}_loader.npz'
                #np.savez(loader_save_path, train_input=train_input, train_target=train_target, val_input=val_input, val_target=val_target)
                #np.savez(loader_save_path, train_loader=train_loader, val_loader=val_loader)
                
                # save loader as train_loader.pickle/val_loader.pickle
                train_loader_save_path = f'{save_path}/train_loader.pickle'
                with open(train_loader_save_path, 'wb') as f:
                    pickle.dump(train_loader, f)
                val_loader_save_path = f'{save_path}/val_loader.pickle'
                with open(val_loader_save_path, 'wb') as f:
                    pickle.dump(val_loader, f)

                #clear the parameter
                del train, val
                del train_input, train_target
                del val_input, val_target
                gc.collect()

        def save_model(model: tf.keras.Model, input_shape, num_classes, save_path):
            #################### model ####################
            # (load) model
            keras_model = model
            # input_shape like (None, 32, 32, 3)
            input_shape = (None,) + input_shape
            print (f'input_shape={input_shape}')
            # tensorflow-onnx(維度可從dataset獲取)
            keras_model_spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
            model_proto, external_tensor_storage = tf2onnx.convert.from_keras(keras_model,
                        input_signature=keras_model_spec, opset=None, custom_ops=None,
                        custom_op_handlers=None, custom_rewriter=None,
                        inputs_as_nchw=None, extra_opset=None, shape_override=None,
                        target=None, large_model=False, output_path=None)
            onnx_model = model_proto
            
            # Save the ONNX model
            onnx_model_path = f'{save_path}/model.onnx'
            onnx.save(onnx_model, onnx_model_path)
            
            #clear the parameter
            del onnx_model, model_proto, external_tensor_storage, keras_model_spec, keras_model, model
            gc.collect()

        def wait_metrics(num_batch, save_path, num_classes):
            print (f'num_batch={num_batch}')
            print (f'num_classes={num_classes}')
            timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.localtime(time.time())))
            input_finish_info_path = f'{save_path}/input_finish_info.npz'
            np.savez(input_finish_info_path, num_batch=num_batch, num_classes=num_classes , timestamp=timestamp)

            #check metrics exsit
            metrics_finish_info_path = f'{save_path}/metrics_finish_info.npz'
            while not os.path.isfile(metrics_finish_info_path):
                time.sleep(5)
                print (f'wait for metrics_finish_info')
            
            print (f'find metrics_finish_info')
            time.sleep(5)
            os.remove(metrics_finish_info_path)

            #load metrics(metrics_ntk)
            '''
            ntks_mses_save_path = f'{save_path}/ntks_mses.npz'
            metrics = np.load(ntks_mses_save_path)
            ntks = metrics['ntks']
            mses = metrics['mses']
            '''

            #load metrics(metrics_ntk_v2)
            ntks_save_path = f'{save_path}/ntks.npz'
            metrics = np.load(ntks_save_path)
            ntks = metrics['ntks']

            return ntks

        # save dataset
        save_dataset(dataset, batch_size, input_shape, num_classes, num_batch, save_path)

        # save model
        save_model(model, input_shape, num_classes, save_path)
        
        # wait ntk
        ntks = wait_metrics(num_batch, save_path, num_classes)

        return ntks

