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
        save_path_list = os.path.listdir(save_path)
        for full_filename in save_path_list:
            if "input_finish_info" in full_filename:
                filename = full_filename

    #check input_finish_info.pickle exsit
    input_finish_info_save_path = f'{save_path}/{filename}.pickle'
    while not os.path.isfile(input_finish_info_save_path):
        time.sleep(10)
        # check the timestamp
        print (f'wait for input_finish_info')
    
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
        onnx_model_path = onnx_model_path = f'{save_path}/model_{timestamp}.onnx'
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
    def get_ntk_n(loader, networks, train_mode=True, num_batch=1, num_classes=10): 
        print (f'num_batch={num_batch}')
        print (f'num_classes={num_classes}')       
        #################### ntk ####################
        device = torch.cuda.current_device()
        # if recalbn > 0:
        #     network = recal_bn(network, xloader, recalbn, device)
        #     if network_2 is not None:
        #         network_2 = recal_bn(network_2, xloader, recalbn, device)
        ntks = []
        for network in networks:
            if train_mode:
                network.train()
            else:
                network.eval()
        ######
        # 建立grads list，將裡面有數量為networks list長度的空list
        grads = [[] for _ in range(len(networks))]
        # xloader 內有 inputs和targets
        for i, (inputs, targets) in enumerate(loader):
            # num_batch 預設為-1
            if num_batch > 0 and i >= num_batch: break
            # numpy to pytorch tensor
            # torch.cuda.DoubleTensor to torch.cuda.FloatTensor
            inputs = torch.from_numpy(inputs).float()
            targets = torch.from_numpy(targets)
            # 將inputs, targets放入gpu
            print (f'inputs={inputs.shape}')
            inputs = inputs.to(device, non_blocking=True)
            # 對networks list內的每個network
            for net_idx, network in enumerate(networks):
                # 將network(weight)放入gpu
                network.to(device)
                # 將network的梯度歸零
                network.zero_grad()
                # 會將梯度疊加給inputs_
                inputs_ = inputs.clone().to(device, non_blocking=True)
                # logit 是inputs_作為輸入的netowrk輸出
                logit = network(inputs_)
                # 若logit 是tuple的話
                if isinstance(logit, tuple):
                    # 則logit 將變成logit tuple第二個元素
                    logit = logit[1]  # 201 networks: return features and logits
                # 將每個inputs_送進network中，並將梯度輸出放在logit list內
                for _idx in range(len(inputs_)):
                    # 對於在inputs_中的每個input，傳入和輸出一樣shape且全為1的矩陣，可得到所有子結點的梯度
                    logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                    # 建立梯度list
                    grad = []
                    # 對所有netowrk的參數
                    for name, W in network.named_parameters():
                        
                        # 將權重梯度append進grad中
                        if 'weight' in name and W.grad is not None:
                            grad.append(W.grad.view(-1).detach())
                    # 再將grad放進grads list中
                    grads[net_idx].append(torch.cat(grad, -1))
                    # 將network梯度歸零
                    network.zero_grad()
                    # 清空cache
                    torch.cuda.empty_cache()
        #pdb.set_trace()
        ######
        # 
        grads = [torch.stack(_grads, 0) for _grads in grads]
        ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
        conds = []
        for ntk in ntks:
            eigenvalues, _ = torch.symeig(ntk)  # ascending
            conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
        return conds

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
        onnx_model_path = onnx_model_path = f'{save_path}/model_rn_{timestamp}.onnx'
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

