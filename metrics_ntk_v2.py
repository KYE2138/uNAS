import ray
#################### TEGNAS testntk #################### 
import numpy as np
import torch
import torch.nn as nn
import tf2onnx
import onnx
import onnx2torch
import pdb
import gc
import os
import time
#################### GPU ####################
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

ray.init(num_cpus=2, num_gpus=1)

@ray.remote(num_gpus=1, max_calls=1)
def get_ntk(num_batch=1, networks_num=3):
    save_path = './tmp/metrics'
    num_batch = num_batch
    timestamp = ''
    networks_num = networks_num
    num_classes = 10
    ntks = -1

    # return timestamp, num_batch
    def wait_input(save_path):
        input_finish_info_path = f'{save_path}/input_finish_info.npz'
        while not os.path.isfile(input_finish_info_path):
            time.sleep(5)
            print (f'wait for input_finish_info')
        print (f'find input_finish_info')
        time.sleep(5)
        #load input_finish_info and del
        input_finish_info = np.load(input_finish_info_path)
        num_batch = int(input_finish_info['num_batch'])
        num_classes = int(input_finish_info['num_classes'])
        timestamp = str(input_finish_info['timestamp'])
        os.remove(input_finish_info_path)

        return num_batch, num_classes, timestamp

    # return (train_loader, val_loader)
    def load_dataset(save_path, num_batch):
        #################### dataset ####################
        #load loader
        loader_save_path = f'{save_path}/loader.npz'
        loader = np.load(loader_save_path)
        '''
        train_input = loader['train_input']
        train_target = loader['train_target']
        val_input = loader['val_input']
        val_target = loader['val_target']
        '''
        train_loader = loader['train_loader']
        val_loader = loader['val_loader']

        # for get ntk loader input
        '''
        train_loader = []
        val_loader = []
        for i in range(num_batch):
            train_loader.append((train_input,train_target))
            val_loader.append((val_input,val_target))
        '''

        #clear the parameter
        del train_input, train_target
        del val_input, val_target
        gc.collect()
        
        return train_loader, val_loader

    # return model (pytorch)
    def transfer_init_model(save_path):
        #################### model ####################
        # load onnx_model
        onnx_model_path = onnx_model_path = f'{save_path}/model.onnx'
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

    def save_ntks(save_path, ntks):
        # save ntks as ntks.npz
        ntks_save_path = f'{save_path}/ntks.npz'
        np.savez(ntks_save_path, ntks=ntks)

        return


    # wait input
    num_batch, num_classes, timestamp = wait_input(save_path)

    # loaddataset
    train_loader, val_loader = load_dataset(save_path, num_batch)
    
    # transfer and init model
    networks = []
    for i in range(networks_num):
        # transfer and init model
        torch_model = transfer_init_model(save_path)
        networks.append(torch_model)
    
    

    # get ntk_n
    ntks = get_ntk_n(loader=train_loader, networks=networks, train_mode=True, num_batch=1, num_classes=num_classes)

    # save_ntks
    save_ntks(save_path, ntks)

    # save metrics_finish_info
    timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.localtime(time.time())))
    metrics_finish_info_path = f'{save_path}/metrics_finish_info.npz'
    np.savez(metrics_finish_info_path, timestamp=timestamp)

    return


while True:
    ntks = ray.get(get_ntk.remote(networks_num=3, num_batch=1))
