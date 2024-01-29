import numpy as np
import os
import heartpy as hp
import copy
from scipy.signal import stft
import pywt
import torch
import random
import torchvision.transforms as transforms
from utils import normalize, beat_normalize
from torch.utils.data import Dataset
 
class TrainSet(Dataset):
    def __init__(self, folder):
        self.train_data = np.load(os.path.join(folder, 'train.npy'))

    def checkR(self, ecg):
        working_data, measures = hp.process(ecg, 500.0)
        peak_list = working_data['peaklist']
        return peak_list
    
    def __len__(self):
        return self.train_data.shape[0]
 
    def __getitem__(self, index):
        time_instance = self.train_data[index]
        time_instance = time_instance[100:4900,:] #(4800, 12)
       
        # Short Time Fast Fourier Transform
        # 500 is the sample rate of PTB-XL, 360 is the sample rate of MIT-BIH
        f,t, Zxx = stft(time_instance.transpose(1,0),fs=500, window='hann',nperseg=125)
        spectrogram_instance = np.abs(Zxx)  #(12, 63, 78)
        spectrogram_instance = spectrogram_instance.transpose(1,2,0)     #(63, 78, 12)
        return time_instance, spectrogram_instance
 
 
 
class TestSet(Dataset):
    def __init__(self, folder):
        self.test_data = np.load(os.path.join(folder, 'test.npy'))
       
    def __len__(self):
        return self.test_data.shape[0]
    
    def checkR(self, ecg):
        working_data, measures = hp.process(ecg, 500.0)
        peak_list = working_data['peaklist']
        return np.array(peak_list)
   
    def __getitem__(self, index):
        time_instance = self.test_data[index]
        r_index = self.checkR(time_instance[:,1])
        time_instance = time_instance[100:4900,:]
        # Short Time Fast Fourier Transform
        f,t, Zxx = stft(time_instance.transpose(1,0),fs=500, window='hann',nperseg=125)
        spectrogram_instance = np.abs(Zxx)  #(12, 63, 78)
        spectrogram_instance = spectrogram_instance.transpose(1,2,0)     #(63, 78, 12)
        return time_instance, spectrogram_instance, r_index
 
