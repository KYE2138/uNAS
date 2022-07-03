'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

This file is modified from:
https://github.com/VITA-Group/TENAS
'''

import ray
#################### TEGNAS testntk #################### 
import numpy as np
import torch
import torch.nn as nn
import tf2onnx
import onnx
import onnx2torch
import pickle
import pdb
import gc
import os
import time
#################### GPU ####################
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

ray.init(num_cpus=2, num_gpus=1)

@ray.remote(num_gpus=1, max_calls=1)
def wait_input(save_path):
    #get filename
    filename = None
    while filename==None:
        save_path_list = os.listdir(save_path)
        for full_filename in save_path_list:
            if "input_finish_info" in full_filename:
                filename = full_filename
        time.sleep(5)
        # check the timestamp
        print (f'wait for input_finish_info')

    #check input_finish_info.pickle exsit
    input_finish_info_save_path = f'{save_path}/{filename}'
    
    print (f'find input_finish_info')
    time.sleep(5)
    # load input_finish_info.pickle
    with open(input_finish_info_save_path, 'rb') as f:
        input_finish_info = pickle.load(f)
    # del
    os.remove(input_finish_info_save_path)

    return input_finish_info

    
@ray.remote(num_gpus=1, max_calls=1)
def get_ntk(save_path, input_finish_info={}):
    print (f'##########get_ntk##########')
    save_path = save_path
    num_batch = input_finish_info["num_batch"]
    num_classes = input_finish_info["num_classes"]
    num_networks = input_finish_info["num_networks"]
    timestamp = input_finish_info["timestamp"]
    print (f'num_batch={num_batch}')
    print (f'num_classes={num_classes}')
    print (f'num_networks={num_networks}')
    print (f'timestamp={timestamp}')
    
    #get_ntk_n
    num_networks = input_finish_info["num_networks"]
    num_classes = input_finish_info["num_classes"]
    num_batch = input_finish_info["num_batch"]
    ntks = -1

    # return (train_loader, val_loader)
    def load_dataset(save_path):
        #################### dataset ####################

        # train_loader.pickle/val_loader.pickle
        train_loader_save_path = f'{save_path}/train_loader.pickle'
        with open(train_loader_save_path, 'rb') as f:
            train_loader = pickle.load(f)
        val_loader_save_path = f'{save_path}/val_loader.pickle'
        with open(val_loader_save_path, 'rb') as f:
            val_loader = pickle.load(f)
        
        

        #clear the parameter
        #del train_input, train_target
        #del val_input, val_target
        gc.collect()
        
        return train_loader, val_loader

    # return model (pytorch)
    def transfer_init_model(save_path, timestamp):
        #################### model ####################
        # load onnx_model
        onnx_model_path = f'{save_path}/model_{timestamp}.onnx'
        onnx_model = onnx.load(onnx_model_path)

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

        def init_model(model, method='kaiming_norm_fanout'):
            if method == 'kaiming_norm_fanin':
                model.apply(kaiming_normal_fanin_init)
            elif method == 'kaiming_norm_fanout':
                model.apply(kaiming_normal_fanout_init)
            return model

        model = init_model (torch_model)
        
        #clear the parameter
        del onnx_model
        del torch_model
        gc.collect()

        return model

    # return (conds)
    def get_ntk_n(loader, networks, loader_val=None, train_mode=True, num_batch=1, num_classes=10): 
        print (f'num_batch={num_batch}')
        print (f'num_classes={num_classes}')       
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
            print (f'inputs={inputs.shape}')
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
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
                inputs_ = inputs.clone().to(device, non_blocking=True)
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
                    cellgrad = torch.cat(cellgrad, -1) if len(cellgrad) > 0 else torch.Tensor([0]).to(device)
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

                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                #targets_onehot = tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]], device='cuda:0')
                targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
                #targets_onehot_mean = tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')
                targets_onehot_mean = targets_onehot - targets_onehot.mean(0)
                targets_y_onehot_mean.append(targets_onehot_mean)
                for net_idx, network in enumerate(networks):
                    network.zero_grad()
                    # 將network(weight)放入gpu
                    network.to(device)      
                    inputs_ = inputs.clone().to(device, non_blocking=True)
                    logit = network(inputs_)
                    if isinstance(logit, tuple):
                        logit = logit[1]  # 201 networks: return features and logits
                    for _idx in range(len(inputs_)):
                        logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                        cellgrad = []
                        for name, W in network.named_parameters():
                            if 'weight' in name and W.grad is not None and "cell" in name:
                                cellgrad.append(W.grad.view(-1).detach())
                        cellgrad = torch.cat(cellgrad, -1) if len(cellgrad) > 0 else torch.Tensor([0]).to(device)
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

    # loaddataset
    train_loader, val_loader = load_dataset(save_path=save_path)

    # transfer and init model
    networks = []
    for i in range(num_networks):
        # transfer and init model
        torch_model = transfer_init_model(save_path=save_path, timestamp=timestamp)
        networks.append(torch_model) 

    # get ntk_n
    ntks = get_ntk_n(loader=train_loader, networks=networks, num_classes=num_classes, num_batch=num_batch, train_mode=True)

    #del .onnx
    onnx_model_path = f'{save_path}/model_{timestamp}.onnx'
    os.remove(onnx_model_path)

    return ntks

@ray.remote(num_gpus=1, max_calls=1)
def get_rn(save_path, input_finish_info):
    print (f'##########get_rn##########')
    save_path = save_path
    num_batch = input_finish_info["num_batch"]
    num_classes = input_finish_info["num_classes"]
    num_networks = input_finish_info["num_networks"]
    timestamp = input_finish_info["timestamp"]
    print (f'num_batch={num_batch}')
    print (f'num_classes={num_classes}')
    print (f'num_networks={num_networks}')
    print (f'timestamp={timestamp}')

    #compute_RN_score
    num_batch = input_finish_info["num_batch"]

    # return (train_loader, val_loader)
    def load_dataset(save_path):
        #################### dataset ####################

        # train_loader.pickle/val_loader.pickle
        train_loader_save_path = f'{save_path}/train_loader.pickle'
        with open(train_loader_save_path, 'rb') as f:
            train_loader = pickle.load(f)
        val_loader_save_path = f'{save_path}/val_loader.pickle'
        with open(val_loader_save_path, 'rb') as f:
            val_loader = pickle.load(f)
        
        

        #clear the parameter
        #del train_input, train_target
        #del val_input, val_target
        gc.collect()
        
        return train_loader, val_loader

    # return model (pytorch)
    def transfer_init_model_rn(save_path, timestamp):
        #################### model ####################
        # load onnx_model
        onnx_model_path = f'{save_path}/model_rn_{timestamp}.onnx'
        onnx_model = onnx.load(onnx_model_path)

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

        def init_model(model, method='kaiming_norm_fanout'):
            if method == 'kaiming_norm_fanin':
                model.apply(kaiming_normal_fanin_init)
            elif method == 'kaiming_norm_fanout':
                model.apply(kaiming_normal_fanout_init)
            return model

        model = init_model (torch_model)
        #print(model)
        
        #clear the parameter
        del onnx_model
        del torch_model
        gc.collect()

        return model

    class LinearRegionCount(object):
        """Computes and stores the average and current value"""
        def __init__(self, n_samples, gpu=None):
            self.ActPattern = {}
            self.n_LR = -1
            self.n_samples = n_samples
            self.ptr = 0
            self.activations = None
            self.gpu = gpu


        @torch.no_grad()
        def update2D(self, activations):
            n_batch = activations.size()[0]
            n_neuron = activations.size()[1]
            self.n_neuron = n_neuron
            if self.activations is None:
                self.activations = torch.zeros(self.n_samples, n_neuron)
                if self.gpu is not None:
                    self.activations = self.activations.cuda(self.gpu)
            self.activations[self.ptr:self.ptr+n_batch] = torch.sign(activations)  # after ReLU
            self.ptr += n_batch

        @torch.no_grad()
        def calc_LR(self):
            res = torch.matmul(self.activations.half(), (1-self.activations).T.half())
            res += res.T
            res = 1 - torch.sign(res)
            res = res.sum(1)
            res = 1. / res.float()
            self.n_LR = res.sum().item()
            del self.activations, res
            self.activations = None
            if self.gpu is not None:
                torch.cuda.empty_cache()

        @torch.no_grad()
        def update1D(self, activationList):
            code_string = ''
            for key, value in activationList.items():
                n_neuron = value.size()[0]
                for i in range(n_neuron):
                    if value[i] > 0:
                        code_string += '1'
                    else:
                        code_string += '0'
            if code_string not in self.ActPattern:
                self.ActPattern[code_string] = 1

        def getLinearReginCount(self):
            if self.n_LR == -1:
                self.calc_LR()
            return self.n_LR    

    class Linear_Region_Collector:
        def __init__(self, models=[], input_size=(), gpu=None,
                    sample_batch=1, dataset=None, data_path=None, seed=0):
            self.models = []
            self.input_size = input_size  # BCHW
            self.sample_batch = sample_batch
            # self.input_numel = reduce(mul, self.input_size, 1)
            self.interFeature = []
            self.dataset = dataset
            self.data_path = data_path
            self.seed = seed
            self.gpu = gpu
            self.device = torch.device('cuda:{}'.format(self.gpu)) if self.gpu is not None else torch.device('cpu')
            # print('Using device:{}'.format(self.device))

            self.reinit(models, input_size, sample_batch, seed)


        def reinit(self, models=None, input_size=None, sample_batch=None, seed=None):
            if models is not None:
                assert isinstance(models, list)
                del self.models
                self.models = models
                for model in self.models:
                    self.register_hook(model)
                self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch, gpu=self.gpu) for _ in range(len(models))]
            if input_size is not None or sample_batch is not None:
                if input_size is not None:
                    self.input_size = input_size  # BCHW
                    # self.input_numel = reduce(mul, self.input_size, 1)
                if sample_batch is not None:
                    self.sample_batch = sample_batch
                # if self.data_path is not None:
                #     self.train_data, _, class_num = get_datasets(self.dataset, self.data_path, self.input_size, -1)
                #     self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.input_size[0], num_workers=16, pin_memory=True, drop_last=True, shuffle=True)
                #     self.loader = iter(self.train_loader)
            if seed is not None and seed != self.seed:
                self.seed = seed
                torch.manual_seed(seed)
                if self.gpu is not None:
                    torch.cuda.manual_seed(seed)
            del self.interFeature
            self.interFeature = []
            if self.gpu is not None:
                torch.cuda.empty_cache()

        def clear(self):
            self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(self.models))]
            del self.interFeature
            self.interFeature = []
            if self.gpu is not None:
                torch.cuda.empty_cache()

        def register_hook(self, model):
            for m in model.modules():
                if isinstance(m, nn.ReLU):
                    m.register_forward_hook(hook=self.hook_in_forward)

        def hook_in_forward(self, module, input, output):
            if isinstance(input, tuple) and len(input[0].size()) == 4:
                self.interFeature.append(output.detach())  # for ReLU

        def forward_batch_sample(self):
            for _ in range(self.sample_batch):
                # try:
                #     inputs, targets = self.loader.next()
                # except Exception:
                #     del self.loader
                #     self.loader = iter(self.train_loader)
                #     inputs, targets = self.loader.next()
                inputs = torch.randn(self.input_size, device=self.device)

                for model, LRCount in zip(self.models, self.LRCounts):
                    #RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
                    #move model to cuda
                    model.to(device=self.device)
                    self.forward(model, LRCount, inputs)
            return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]

        def forward(self, model, LRCount, input_data):
            self.interFeature = []
            with torch.no_grad():
                # model.forward(input_data.cuda())
                model.forward(input_data)
                if len(self.interFeature) == 0: return
                #RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) 
                feature_data = torch.cat([f.reshape(input_data.size(0), -1) for f in self.interFeature], 1)
                LRCount.update2D(feature_data)

    def compute_RN_score(model: nn.Module, gpu=0, loader=[], num_batch=32):
        
        # loader = [(inputs, targets),(inputs, targets)...]
        # get input_size
        inputs = loader[0][0]
        input_size = inputs.shape
        #fix rns is always =batch_size=64
        input_size=(3000, 2, 2, 1)
        print (f'input_size={input_size}')
        

        lrc_model = Linear_Region_Collector(models=model, input_size=input_size,
                                            gpu=gpu, sample_batch=num_batch)              
        
        #num_linear_regions = float(lrc_model.forward_batch_sample()[0])
        try:
            num_linear_regions = lrc_model.forward_batch_sample()
        except AttributeError:
            num_linear_regions= [0,0,0]
            print("Oops!  Linear_Region_Collector.forward_batch_sample() has AttributeError error, num_linear_regions = [0,0,0]")
        
        del lrc_model
        torch.cuda.empty_cache()
        return num_linear_regions
    
    # loaddataset
    train_loader, val_loader = load_dataset(save_path=save_path)

    # transfer and init model
    networks = []
    for i in range(num_networks):
        # transfer and init model
        torch_model = transfer_init_model_rn(save_path, timestamp)
        networks.append(torch_model) 

    rns = compute_RN_score(model=networks, loader=train_loader, num_batch=num_batch)
    #pdb.set_trace()
    
    #del .onnx
    onnx_model_path = f'{save_path}/model_rn_{timestamp}.onnx'
    os.remove(onnx_model_path)

    return rns

@ray.remote(num_gpus=1, max_calls=1)
def save_metrics(save_path, input_finish_info, ntks, rns):
    timestamp = input_finish_info["timestamp"]
    metrics_finish_info = {"ntks":ntks, "rns":rns, "timestamp":timestamp}
    # save metrics_finish_info as metrics_finish_info.pickle
    metrics_finish_info_save_path = f'{save_path}/metrics_finish_info_{timestamp}.pickle'
    with open(metrics_finish_info_save_path, 'wb') as f:
        pickle.dump(metrics_finish_info, f)
    
    return 1



# main
save_path = './tmp/metrics/ntk_rn'
while True:
    input_finish_info = ray.get(wait_input.remote(save_path=save_path))
    ntks = ray.get(get_ntk.remote(save_path=save_path, input_finish_info=input_finish_info))
    rns = ray.get(get_rn.remote(save_path=save_path, input_finish_info=input_finish_info))
    finish = ray.get(save_metrics.remote(save_path=save_path, input_finish_info=input_finish_info, ntks=ntks, rns=rns))

