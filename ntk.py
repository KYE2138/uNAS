#################### uNAS model_trainer ####################
import logging
from typing import Optional

import tensorflow as tf

from config import TrainingConfig
from pruning import DPFPruning
from utils import debug_mode

#################### TEGNAS testntk #################### 
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import tf2onnx
import onnx
import onnx2torch

import pdb
import gc

#################### GPU ####################
# GPU mem issue
#config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
#sess = tf.compat.v1.Session(config=config)


class ModelNTK:
    """Keras models according to the specified config."""
    def __init__(self, data):
        self.dataset = data

    def get_ntk(self, model: tf.keras.Model, networks_num=3, batch_num=1, batch_size=128):
        dataset = self.dataset
        input_shape = self.dataset.input_shape
        num_classes = self.dataset.num_classes
        batch_size = batch_size
        model = model
        networks_num = networks_num
        batch_num = batch_num

        # gpu
        device = torch.cuda.current_device()
        print (device)
        
        # return (train_loader, val_loader)
        def generate_dataset(dataset, batch_size, input_shape, num_classes, batch_num):
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
            for i in range(batch_num):
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
            
            #clear the parameter
            del train, val
            del train_input, train_target
            del val_input, val_target
            gc.collect()

            return train_loader, val_loader
        # return model (pytorch)
        def transfer_init_model(model: tf.keras.Model, input_shape, num_classes):
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

            # onnx2torch
            torch_model = onnx2torch.convert(onnx_model)

            # Model class must be defined somewhere
            torch_model.eval()

            # torch_model 參數初始化
            def kaiming_normal_fanin_init(m):
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0.0)

            def kaiming_normal_fanout_init(m):
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0.0)

            def init_model(model, method='kaiming_norm_fanin'):
                if method == 'kaiming_norm_fanin':
                    model.apply(kaiming_normal_fanin_init)
                elif method == 'kaiming_norm_fanout':
                    model.apply(kaiming_normal_fanout_init)
                return model

            model = init_model (torch_model)
            
            #clear the parameter
            del keras_model
            del onnx_model
            del torch_model
            gc.collect()

            return model
        # return (conds_x, prediction_mses)
        def get_ntk_n(loader, networks, loader_val, train_mode=True, num_batch=1, num_classes=10):        
            #################### ntk ####################
            device = torch.cuda.current_device()
            ntks = []
            for network in networks:
                if train_mode:
                    network.train()
                else:
                    network.eval()
            ######
            # 建立grads list，長度同networks list
            grads_x = [[] for _ in range(len(networks))]
            # 建立cellgrads_x list，長度同networks list # 建立cellgrads_y list，長度同networks list
            cellgrads_x = [[] for _ in range(len(networks))]; cellgrads_y = [[] for _ in range(len(networks))]
            # For mse
            ntk_cell_x = []; ntk_cell_yx = []; prediction_mses = []
            targets_x_onehot_mean = []; targets_y_onehot_mean = []
            # 對每組inputs和targets
            # inputs = torch.Size([64, 3, 32, 32])
            # targets = torch.Size([64])
            # len(loader) = 1
            # loader = [(inputs, targets)]

            #loader = torch.from_numpy(loader)
            for i, (inputs, targets) in enumerate(loader):
                # num_batch 預設為64
                if num_batch > 0 and i >= num_batch: break
                # numpy to pytorch tensor
                # torch.cuda.DoubleTensor to torch.cuda.FloatTensor
                inputs = torch.from_numpy(inputs).float()
                targets = torch.from_numpy(targets)
                # 將inputs, targets放入gpu
                inputs = inputs.cuda(device=device, non_blocking=True)
                targets = targets.cuda(device=device, non_blocking=True)
                # For mse
                targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
                targets_onehot_mean = targets_onehot - targets_onehot.mean(0)
                targets_x_onehot_mean.append(targets_onehot_mean)
                # 對每個network
                for net_idx, network in enumerate(networks):
                    # 將network(weight)放入gpu
                    network.to(device)
                    # 將network的梯度歸零
                    network.zero_grad()
                    # 會將梯度疊加給inputs_
                    inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
                    # logit 是inputs_作為輸入的netowrk輸出, logit = (64, 10)
                    logit = network(inputs_)
                    # 若logit 是tuple的話(for nasbach201)
                    if isinstance(logit, tuple):
                        logit = logit[1]  # 201 networks: return features and logits
                    # _idx = 0~63 ,inputs_ = (64, 32, 32, 3) for cifar10
                    for _idx in range(len(inputs_)):
                        # batch=64, logit = (64, 10), logit[_idx:_idx+1] = (10)
                        # 計算各個Variable的梯度，調用根節點variable的backward方法，autograd會自動沿著計算圖反向傳播，計算每一個葉子節點的梯度
                        # Grad_variables：形狀與variable一致，對於logit[_idx:_idx+1].backward()，指定logit[_idx:_idx+1]有哪些要計算梯度
                        # Retain_graph：反向傳播需要緩存一些中間結果，反向傳播之後，這些緩存就被清空，可通過指定這個參數不清空緩存，用來多次反向傳播
                        logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                        # 建立梯度list, 用來存放network中的W
                        # W分為grad和cellgrad
                        grad = []
                        cellgrad = []
                        # 在預設的TinyNetworkDarts Class中，named_parameters()會獲得model中所有參數的名稱
                        for name, W in network.named_parameters():
                            # 在name中有weight('Conv_5.weight')的W.grad，append進grad list
                            if 'weight' in name and W.grad is not None:
                                # 將W.grad resize成1維，並複製(不在計算圖中)
                                grad.append(W.grad.view(-1).detach())
                                # 在name中有cells('cells.0.edges.1<-0.3.op.1.weight')的W.grad，append進grad list
                                if "cell" in name:
                                    cellgrad.append(W.grad.view(-1).detach())                        
                        # 將(單個network的)grad list [tensor (64, 8148)]轉換成tensor (64, 8148)，append進grads_x list
                        grads_x[net_idx].append(torch.cat(grad, -1)) 
                        cellgrad = torch.cat(cellgrad, -1) if len(cellgrad) > 0 else torch.Tensor([0]).cuda()
                        if len(cellgrads_x[net_idx]) == 0:
                            cellgrads_x[net_idx] = [cellgrad]
                        else:
                            cellgrads_x[net_idx].append(cellgrad)
                        network.zero_grad()
                        torch.cuda.empty_cache()
                    '''
                    # del cuda tensor
                    del network, inputs_, inputs, targets, cellgrad
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    '''

            # For MSE, 將targets_x_onehot_mean list [tensor (64, 10)]轉換成tensor (64, 10)
            #torch.Size([64, 10])
            targets_x_onehot_mean = torch.cat(targets_x_onehot_mean, 0)

            # cell's NTK #####
            for _i, grads in enumerate(cellgrads_x):
                grads = torch.stack(grads, 0)
                _ntk = torch.einsum('nc,mc->nm', [grads, grads])
                ntk_cell_x.append(_ntk)
                cellgrads_x[_i] = grads
            # NTK cond
            grads_x = [torch.stack(_grads, 0) for _grads in grads_x]
            ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads_x]
            conds_x = []
            # ntk = torch.Size([64, 64])
            # len(ntks) = 3
            for ntk in ntks:
                eigenvalues, _ = torch.symeig(ntk)  # ascending
                _cond = eigenvalues[-1] / eigenvalues[0]
                if torch.isnan(_cond):
                    conds_x.append(-1) # bad gradients
                else:
                    conds_x.append(_cond.item())

            # Val / Test set
            if loader_val is not None:
                for i, (inputs, targets) in enumerate(loader_val):
                    if num_batch > 0 and i >= num_batch: break
                    # numpy to pytorch tensor
                    # torch.cuda.DoubleTensor to torch.cuda.FloatTensor
                    inputs = torch.from_numpy(inputs).float()
                    targets = torch.from_numpy(targets)

                    inputs = inputs.cuda(device=device, non_blocking=True)
                    targets = targets.cuda(device=device, non_blocking=True)
                    #targets_onehot = tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]], device='cuda:0')
                    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
                    #targets_onehot_mean = tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')
                    targets_onehot_mean = targets_onehot - targets_onehot.mean(0)
                    targets_y_onehot_mean.append(targets_onehot_mean)
                    for net_idx, network in enumerate(networks):
                        network.zero_grad()
                        # 將network(weight)放入gpu
                        network.to(device)      
                        inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
                        logit = network(inputs_)
                        if isinstance(logit, tuple):
                            logit = logit[1]  # 201 networks: return features and logits
                        for _idx in range(len(inputs_)):
                            logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                            cellgrad = []
                            for name, W in network.named_parameters():
                                if 'weight' in name and W.grad is not None and "cell" in name:
                                    cellgrad.append(W.grad.view(-1).detach())
                            cellgrad = torch.cat(cellgrad, -1) if len(cellgrad) > 0 else torch.Tensor([0]).cuda()
                            if len(cellgrads_y[net_idx]) == 0:
                                cellgrads_y[net_idx] = [cellgrad]
                            else:
                                cellgrads_y[net_idx].append(cellgrad)
                            network.zero_grad()
                            torch.cuda.empty_cache()
                targets_y_onehot_mean = torch.cat(targets_y_onehot_mean, 0)
                for _i, grads in enumerate(cellgrads_y):
                    grads = torch.stack(grads, 0)
                    # cellgrads_y[0].shape = torch.Size([64, 1])
                    cellgrads_y[_i] = grads
                '''
                # del cuda tensor
                del network, inputs_, inputs, targets, cellgrad
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                '''

                for net_idx in range(len(networks)):
                    try:
                        _ntk_yx = torch.einsum('nc,mc->nm', [cellgrads_y[net_idx], cellgrads_x[net_idx]])
                        PY = torch.einsum('jk,kl,lm->jm', _ntk_yx, torch.inverse(ntk_cell_x[net_idx]), targets_x_onehot_mean)
                        prediction_mses.append(((PY - targets_y_onehot_mean)**2).sum(1).mean(0).item())
                        # clear the parameter
                    except RuntimeError:
                        # RuntimeError: inverse_gpu: U(1,1) is zero, singular U.
                        # prediction_mses.append(((targets_y_onehot_mean)**2).sum(1).mean(0).item())
                        prediction_mses.append(-1) # bad gradients
         

            ######
            if loader_val is None:
                return conds_x
            else:
                return conds_x, prediction_mses

        # generate_dataset
        train_loader, val_loader = generate_dataset(dataset, batch_size, input_shape, num_classes, batch_num)
        # train_loader = [(inputs, targets)]
        # val_loader = [(inputs, targets)]

        # init_transfer_model
        networks = []
        for i in range(networks_num):
            torch_model = transfer_init_model(model, input_shape, num_classes)
            networks.append(torch_model)
        
        # get_ntk_n
        #ntks, mses = get_ntk_n(loader=train_loader, networks=networks, loader_val=val_loader, train_mode=True, num_batch=1, num_classes=10)
        ntks = []
        mses = []
        print ("ntks:",ntks)
        print ("mses:",mses)

        #clear the parameter
        del torch_model, networks
        #torch.cuda.empty_cache()
        #torch.cuda.ipc_collect()

        #pdb.set_trace()
        return ntks

