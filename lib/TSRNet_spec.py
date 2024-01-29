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

#Spectrogram Reconstruction
class TSRNet_spec(nn.Module):

    def __init__(self, enc_in):
        super(TSRNet_spec, self).__init__()

        self.channel = enc_in

        # Spectrogram module
        self.spec_encoder = Encoder2D(enc_in)
        self.spec_decoder = Decoder1D(enc_in+1)
        
        self.conv_spec1 = nn.Conv1d(50*51, 50, 3, 1, 1, bias=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(66, 136),
            nn.LayerNorm(136),
            nn.ReLU()
        )

    def forward(self, spectrogram_ecg):
        #Spectrogram ECG encode
        spectrogram_features = self.spec_encoder(spectrogram_ecg.permute(0,3,1,2)) #(32, 50, 63, 66)
        
        n, c, h, w = spectrogram_features.shape
        spectrogram_features = self.conv_spec1(spectrogram_features.contiguous().view(n, c*h, w)) #(32, 50, 66)
        spectrogram_features = self.mlp(spectrogram_features)

        #Spectrogram ECG decoder
        gen_spectrogram = self.spec_decoder(spectrogram_features)
        output = gen_spectrogram.transpose(-1, 1)

        return (output[:,:,0:self.channel],output[:,:,self.channel:self.channel+1])

