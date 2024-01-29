import os
import random
import argparse
import time
import torch
import math
import numpy as np
from tqdm import tqdm
from utils import time_string, convert_secs2time, AverageMeter, normalize
from lib.TSRNet import TSRNet
from dataloader import TrainSet, TestSet
from sklearn.metrics import roc_auc_score
import copy
import warnings
import torch.nn as nn
 
def main(args):
    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    else:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
       
    print("args: ", args)
   
    # load dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
 
    dset = TrainSet(folder=args.data_path)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True, **kwargs)
 
    dtset = TestSet(folder=args.data_path)
    test_loader = torch.utils.data.DataLoader(dtset, batch_size=1, shuffle=False, **kwargs)
    labels = np.load(os.path.join(args.data_path, 'label.npy'))
 
    # load model
    if args.spec:
        model = TSRNet(enc_in=args.dims).to(device)
    else:
        model = TSRNet_time(enc_in=args.dims).to(device)
       
    optimizer = torch.optim.AdamW(model.parameters() , lr=args.lr, weight_decay=1e-5)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")

    # continue training
    if args.pth_path is not None:
        checkpoint = torch.load(args.pth_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    # start training
    start_time = time.time()
    epoch_time = AverageMeter()
 
    old_auc_result = 0
    for epoch in range(0, args.epochs + 1):
        adjust_learning_rate(optimizer, args.lr, epoch, args)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time))
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
 
        train(args, model, epoch, train_loader, optimizer)
        auc_result = test(args, model, test_loader, labels)
        if auc_result > old_auc_result:
            old_auc_result = auc_result
            if args.save_model == 1:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, args.save_path + 'TSRNet-%d.pt' % epoch)
    print("final best auc: ", old_auc_result)
 
 
def train(args, model, epoch, train_loader, optimizer):
    model.train()
    total_losses = AverageMeter()
    for i, (time_ecg, spectrogram_ecg) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
    
        #time branch
        time_ecg = time_ecg.float().to(device)  #(32, 4800, 12)
        bs, time_length, dim = time_ecg.shape
        mask_time = copy.deepcopy(time_ecg)
        mask = torch.zeros((bs,time_length,1), dtype=torch.bool).to(device)
        patch_length = time_length // 100 #48
        for j in random.sample(range(0,100), args.mask_ratio_time):
            mask[:, j*patch_length:(j+1)*patch_length] = 1 #(32, 48, 1)
        mask_time = torch.mul(mask_time, ~mask)
 
        if args.spec:
            #spectrogram branch
            spec_ecg = spectrogram_ecg.float().to(device) #(32, 63, 66, 12)
            bs, freq_dim, time_dim, dim = spec_ecg.shape
            mask_spec = copy.deepcopy(spec_ecg)
            #add mask to spectrogram ecg
            mask = torch.zeros((bs, freq_dim, time_dim, 1), dtype=torch.bool).to(device)
            patch_length = 1
            for j in random.sample(range(0,66), args.mask_ratio_spec):
                mask[:, :, j*patch_length:(j+1)*patch_length, :] = 1 #(32, 63, 1, 1)
            mask_spec = torch.mul(mask_spec, ~mask)
           
            (gen_time, time_var) = model(mask_time, mask_spec)
            
            #loss time ecg
            time_err = (gen_time - time_ecg) ** 2
            l_time = torch.mean(torch.exp(-time_var)*time_err) + torch.mean(time_var) 
            loss = l_time 
        else:
            (gen_time, time_var) = model(mask_time)
            #loss time ecg
            time_err = (gen_time - time_ecg) ** 2
            l_time = torch.mean(torch.exp(-time_var)*time_err) + torch.mean(time_var) 
            loss = l_time 
           
        loss.backward()
        optimizer.step()
 
        total_losses.update(loss.item(), bs)
       
    print(('Train Epoch: {} Total_Loss: {:.6f}'.format(epoch, total_losses.avg)))
 
 
 
def test(args, model, test_loader, labels):
    torch.zero_grad = True
    model.eval()
    result = []
    for i, (time_ecg, spectrogram_ecg, r_index) in tqdm(enumerate(test_loader)):
        instance_result = []
        # loss mask based on R peaks
        time_length = time_ecg.shape[1]
        idx_length = r_index.shape[1]
        mask_loss = torch.zeros((time_length, args.dims), dtype=torch.bool).to(device)
        for r_idx in range(idx_length):
            r_index_value = r_index[0][r_idx]
            if r_index_value>200 and r_index_value<4800-400:
                left = max(0, r_index_value - 240)
                mask_loss[left:r_index_value+240,:] = 1
 
        for j in range(100//args.mask_ratio_time):
            # mask on time branch
            patch_interval_time = 4800 // args.mask_ratio_time
            time_ecg = time_ecg.float().to(device)
            mask_time = copy.deepcopy(time_ecg)
            mask = torch.zeros((1,time_length,1), dtype=torch.bool).to(device)
           
            for k in range(args.mask_ratio_time):
                cut_idx = 48*j + patch_interval_time*k
                mask[:,cut_idx:cut_idx+48] = 1
            mask_time = torch.mul(mask_time, ~mask)
 
            if args.spec:
                # mask on spectrogram branch
                patch_interval_spec = 66 // args.mask_ratio_spec
                spec_ecg = spectrogram_ecg.float().to(device)
                bs, freq_dim, time_dim, dim = spec_ecg.shape
                mask_spec = copy.deepcopy(spec_ecg)
                #add mask to spectrogram ecg
                mask = torch.zeros((bs, freq_dim, time_dim, 1), dtype=torch.bool).to(device)
 
                for k in range(args.mask_ratio_spec):
                    cut_idx = 1*j + patch_interval_spec*k
                    mask[:, :, cut_idx:cut_idx+1] = 1
                mask_spec = torch.mul(mask_spec, ~mask)
 
                (gen_time, time_var) = model(mask_time, mask_spec)
            else:
                (gen_time, time_var) = model(mask_time)
 
            #loss time ecg
            time_err = (gen_time - time_ecg) ** 2
            if args.mask_loss:
                l_time = torch.exp(-time_var)*time_err
                l_time = torch.mul(l_time, mask_loss)
                l_time = torch.sum(l_time)/torch.sum(mask_loss)
                loss = l_time 
            else:
                l_time = torch.mean(torch.exp(-time_var)*time_err)
                loss = l_time
 
            loss = loss.detach().cpu().numpy()
            instance_result.append(loss)
       
        tmp_instance_result = np.asarray(instance_result)
        result.append(tmp_instance_result.mean())
 
    scores = np.asarray(result)
    test_labels = np.array(labels).astype(int)
 
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
 
    auc_result = roc_auc_score(test_labels, scores)
 
    print(("AUC: ", round(auc_result, 3)))
    return auc_result
 
def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
 
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='ECG Anomaly Detection based on Signal and Spectrogram Restoration')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    parser.add_argument('--dims', type=int, default=12, help='dimension of the input data')
    parser.add_argument('--save_model', type=int, default=1, help='0 for discard, 1 for save model')
    parser.add_argument('--save_path', type=str, default='ckpt/')
    parser.add_argument('--mask_ratio_time', type=int, default=30, help='mask ratio for self-restoration in time branch')
    parser.add_argument('--mask_ratio_spec', type=int, default=20, help='mask ratio for self-restoration in spectrogram branch')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--spec", default=True)
    parser.add_argument("--pth_path", type=str, default=None)
    parser.add_argument("--mask_loss", default=True)  #Peak-based Error

    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = "cuda:" + args.gpu if use_cuda else 'cpu'
    main(args)


