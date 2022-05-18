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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

ray.init(num_cpus=8, num_gpus=1)

@ray.remote(num_gpus=1, max_calls=1)
def get_ntk(num_batch=1, networks_num=3):
    save_path = './tmp/metrics'
    num_batch = num_batch
    timestamp = ''
    networks_num = networks_num
    num_classes = 10
    ntks = -1
    mses = -1

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
        train_input = loader['train_input']
        train_target = loader['train_target']
        val_input = loader['val_input']
        val_target = loader['val_target']

        # for get ntk loader input
        train_loader = []
        val_loader = []
        for i in range(num_batch):
            train_loader.append((train_input,train_target))
            val_loader.append((val_input,val_target))
        
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

        def init_model(model, method='kaiming_norm_fanin'):
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

    def save_ntks_mses(save_path, ntks, mses):
        # save ntks mses as ntks_mses.npz
        ntks_mses_save_path = f'{save_path}/ntks_mses.npz'
        np.savez(ntks_mses_save_path, ntks=ntks, mses=mses)

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
    ntks, mses = get_ntk_n(loader=train_loader, networks=networks, loader_val=val_loader, train_mode=True, num_batch=1, num_classes=num_classes)

    # save_ntks
    save_ntks_mses(save_path, ntks, mses)

    # save metrics_finish_info
    timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.localtime(time.time())))
    metrics_finish_info_path = f'{save_path}/metrics_finish_info.npz'
    np.savez(metrics_finish_info_path, timestamp=timestamp)

    return


while True:
    ntks = ray.get(get_ntk.remote(networks_num=3, num_batch=1))
    pdb.set_trace()