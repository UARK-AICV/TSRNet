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

#Time and Spectrogram Restoration
class TSRNet(nn.Module):

    def __init__(self, enc_in):
        super(TSRNet, self).__init__()

        self.channel = enc_in

        # Time series module 
        self.time_encoder = Encoder1D(enc_in)
        self.time_decoder = Decoder1D(enc_in+1)
        
        # Spectrogram module
        self.spec_encoder = Encoder2D(enc_in)
    
        self.conv_spec1 = nn.Conv1d(50*51, 50, 3, 1, 1, bias=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(202, 136),
            nn.LayerNorm(136),
            nn.ReLU()
        )
        
        self.attn1 = MultiHeadedAttention(2, 50)
        self.drop = nn.Dropout(0.1)
        self.layer_norm1 = LayerNorm(50)

    def attention_func(self,x, attn, norm):
        attn_latent = attn(x, x, x)
        attn_latent = norm(x + self.drop(attn_latent))
        return attn_latent
    
    def forward(self, time_ecg, spectrogram_ecg):
        #Time ECG encode
        time_features = self.time_encoder(time_ecg.transpose(-1,1)) #(32, 50, 136)

        #Spectrogram ECG encode
        spectrogram_features = self.spec_encoder(spectrogram_ecg.permute(0,3,1,2)) #(32, 50, 63, 66)
        n, c, h, w = spectrogram_features.shape
        spectrogram_features = self.conv_spec1(spectrogram_features.contiguous().view(n, c*h, w)) #(32, 50, 66)
        
        latent_combine = torch.cat([time_features, spectrogram_features], dim=-1)
        #Cross-attention
        latent_combine = latent_combine.transpose(-1, 1)
        attn_latent = self.attention_func(latent_combine, self.attn1, self.layer_norm1)
        attn_latent = self.attention_func(attn_latent, self.attn1, self.layer_norm1)
        latent_combine = attn_latent.transpose(-1, 1)
        
        latent_combine = self.mlp(latent_combine)
        
        output = self.time_decoder(latent_combine)
        output = output.transpose(-1, 1)

        return  (output[:,:,0:self.channel],output[:,:,self.channel:self.channel+1])