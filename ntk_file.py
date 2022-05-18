#################### uNAS model_trainer ####################
import logging
from typing import Optional

import tensorflow as tf

from config import TrainingConfig
from pruning import DPFPruning
from utils import debug_mode

#################### save_dataset save_model #################### 
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx

import os
import pdb
import gc

#################### GPU ####################
# GPU mem issue
#config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
#sess = tf.compat.v1.Session(config=config)


class ModelNTKFile:
    """Keras models according to the specified config."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.save_path = "./tmp/ntk_file"

    def save_ntk_input(self, model: tf.keras.Model, num_batch):
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
            # from uNAS dataset by tf
            train = dataset.train_dataset() \
                .shuffle(batch_size * 8) \
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
                #train_loader.append((train_input,train_target))

                # for get ntk val_loader input
                # list(val.as_numpy_iterator())[0][0].shape = (128, 32, 32, 3)
                val_input = list(val.as_numpy_iterator())[0][0]
                # list(val.as_numpy_iterator())[0][1].shape = (128, 1)?
                val_target = list(val.as_numpy_iterator())[0][1]
                # targets.shape = torch.Size([128])
                val_target = val_target.reshape((-1,))
                # one_hot is only applicable to index tensor
                val_target = val_target.astype(np.int64)
                #val_loader.append((val_input,val_target))

            np.save(f'{save_path}/train_input.npy', train_input)
            np.save(f'{save_path}/train_target.npy', train_target)
            np.save(f'{save_path}/val_input.npy', val_input)
            np.save(f'{save_path}/val_target.npy', val_target)

            #clear the parameter
            del train, val
            del train_input, train_target
            del val_input, val_target
            gc.collect()

        def save_model(model: tf.keras.Model, input_shape, num_classes, save_path):
            #################### model ####################
            # (load) model
            keras_model = model

            # tensorflow-onnx(維度可從dataset獲取)
            keras_model_spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)
            model_proto, external_tensor_storage = tf2onnx.convert.from_keras(keras_model,
                        input_signature=keras_model_spec, opset=None, custom_ops=None,
                        custom_op_handlers=None, custom_rewriter=None,
                        inputs_as_nchw=None, extra_opset=None, shape_override=None,
                        target=None, large_model=False, output_path=None)
            onnx_model = model_proto
            
            # Save the ONNX model
            onnx.save(onnx_model, f'{save_path}/model.onnx')
            
            #clear the parameter
            del onnx_model, model_proto, external_tensor_storage, keras_model_spec, keras_model, model
            gc.collect()

        def wait_ntk(num_batch):
            finish_info = np.array([[]])
            finish_info = np.append(finish_info,[[num_batch]])
            finish_info = np.append(finish_info,[[num_batch]])
            np.save(f'{save_path}/finish_info.npy', finish_info)


            

        # save dataset
        save_dataset(dataset, batch_size, input_shape, num_classes, num_batch, save_path)

        # save model
        save_model(model, input_shape, num_classes, save_path)
        
        # wait ntk
        wait_ntk(num_batch)

        pdb.set_trace()
        return True

