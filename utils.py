import time
import random
import numpy as np
import heartpy as hp
import torch
import copy
import math
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def normalize(X_train_ori):
    X_train = copy.deepcopy(X_train_ori)
    for count in range(X_train.shape[0]):
        for j in range(12):
            seq = X_train[count][:,j]
            X_train[count][:,j] = 2*(seq-seq.min())/(seq.max()-seq.min())-1
    return X_train


def beat_normalize(X_train_ori):
    X_train = copy.deepcopy(X_train_ori)
    for j in range(12):
        seq = X_train[:,j]
        X_train[:,j] = 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1
    return X_train