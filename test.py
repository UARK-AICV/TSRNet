import os
import random
import argparse
import time
import math
import torch
import numpy as np
from tqdm import tqdm
from utils import normalize
from lib.TSRNet import TSRNet
from lib.TSRNet_time import TSRNet_time
from dataloader import TestSet
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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

    dtset = TestSet(folder=args.data_path)
    test_loader = torch.utils.data.DataLoader(dtset, batch_size=1, shuffle=False, **kwargs)
    labels = np.load(os.path.join(args.data_path, 'label.npy'))

    # load model
    if args.spec:
        model = TSRNet(enc_in=args.dims).to(device)
    else:
        model = TSRNet_time(enc_in=args.dims).to(device)
    if args.load_model == 1:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    # test
    detection_test(args, model, test_loader, labels)


def detection_test(args, model, test_loader, labels):
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

    print(("Detection AUC: ", round(auc_result, 3)))
    return auc_result



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Testing ECG Anomaly Detection based on Signal and Spectrogram Reconstruction')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--dims', type=int, default=12, help='dimension of the input data')
    parser.add_argument('--load_model', type=int, default=1, help='0 for retrain, 1 for load model')
    parser.add_argument('--load_path', type=str, default='ckpt/TSRNet.pt')
    parser.add_argument('--mask_ratio_time', type=int, default=30, help='mask ratio for self-restoration in time branch')
    parser.add_argument('--mask_ratio_spec', type=int, default=20, help='mask ratio for self-restoration in spectrogram branch')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--spec", default=False)
    parser.add_argument("--mask_loss", default=False)  #Peak-based Error

    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = "cuda:" + args.gpu if use_cuda else 'cpu'
    main(args)




