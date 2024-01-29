import numpy as np
import torch
import torch.nn as nn
import os
import copy
import torch.nn.functional as F
import math
import random
from utils import *
from lib.modules import *


#Time Reconstruction
class TSRNet_time(nn.Module):

    def __init__(self, enc_in):
        super(TSRNet_time, self).__init__()

        self.channel = enc_in

        # Time series module 
        self.time_encoder = Encoder1D(enc_in)
        self.time_decoder = Decoder1D(enc_in+1)

    def forward(self, time_ecg):
        #Time ECG encode
        time_features = self.time_encoder(time_ecg.transpose(-1,1)) #(32, 50, 136)

        #Time ECG decode
        gen_time = self.time_decoder(time_features)
        gen_time = gen_time.transpose(-1, 1)

        return  (gen_time[:,:,0:self.channel],gen_time[:,:,self.channel:self.channel+1])

