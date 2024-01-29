<h1 align="center">TSRNet: Simple Framework for Real-time ECG Anomaly Detection with Multimodal Time and Spectrogram Restoration Network </h1>
<p align="center">
  <p align="center">
    <a href="https://tanbuinhat.github.io/"><strong>Nhat-Tan Bui</strong></a>
    ·
    <a href="https://dblp.org/pid/253/9950.html"><strong>Dinh-Hieu Hoang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=zsGhPHcAAAAJ&hl=vi&authuser=1"><strong>Thinh Phan</strong></a>
    ·
    <a href="https://www.fit.hcmus.edu.vn/~tmtriet/"><strong>Minh-Triet Tran</strong></a>
    .
    <a href="https://directory.hsc.wvu.edu/Profile/60996"><strong>Brijesh Patel</strong></a>
    .
    <a href="https://community.wvu.edu/~daadjeroh/"><strong>Donald Adjeroh</strong></a>
    .
    <a href="https://www.nganle.net/"><strong>Ngan Le</strong></a>
  </p>

  <h4 align="center"><a href="https://arxiv.org/abs/2312.10187">arXiv</a></h4>
  <div align="center"></div>

</p>

## Introduction
<image src="Images/main_architecture.png">
  
The electrocardiogram (ECG) is a valuable signal used to assess various aspects of heart health, such as heart rate and rhythm. It plays a crucial role in identifying cardiac conditions and detecting anomalies in ECG data. However, distinguishing between normal and abnormal ECG signals can be a challenging task. In this paper, we propose an approach that leverages anomaly detection to identify unhealthy conditions using solely normal ECG data for training. Furthermore, to enhance the information available and build a robust system, we suggest considering both the time series and time-frequency domain aspects of the ECG signal. As a result, we introduce a specialized network called the Multimodal Time and Spectrogram Restoration Network (TSRNet) designed specifically for detecting anomalies in ECG signals. TSRNet falls into the category of restoration-based anomaly detection and draws inspiration from both the time series and spectrogram domains. By extracting representations from both domains, TSRNet effectively captures the comprehensive characteristics of the ECG signal. This approach enables the network to learn robust representations with superior discrimination abilities, allowing it to distinguish between normal and abnormal ECG patterns more effectively. Furthermore, we introduce a novel inference method, termed Peak-based Error, that specifically focuses on ECG peaks, a critical component in detecting abnormalities. The experimental result on the large-scale dataset PTB-XL has demonstrated the effectiveness of our approach in ECG anomaly detection, while also prioritizing efficiency by minimizing the number of trainable parameters.
<table border="0">
  <tr>
    <td><image src="Images/cross_attention.png"></td>
    <td><image src="Images/peak_based.png"></td>
  </tr>
</table>

## Prerequisites
<ul>
  <li>Pytorch</li>
  <li>Torchvision</li>
  <li>Numpy</li>
  <li>SciPy</li>
  <li>HeartPy</li>
  <li>PyWavelets</li>
</ul>

## Datasets
To validate the effectiveness of our model, we conduct the experiments benchmark in PTB-XL dataset. We follow the same dataset preprocessing as in <a href="https://github.com/MediaBrain-SJTU/ECGAD">Jiang et al.</a> Please refer to their repository for more details.</a>

## Usage

### Training
<ul>
  <li>Without Spectrogram branch</li>
  
```
python train.py --dims "dimension of ECG signal" --data_path "path to train dataset" --save_path "path to save checkpoint"
```

  <li>With Spectrogram branch</li>
  
```
python train.py --dims "dimension of ECG signal" --data_path "path to train dataset" --save_path "path to save checkpoint" --spec
```
</ul>

### Testing
<ul>
  <li>Without Peak-based Error</li>
  
```
python test.py --dims "dimension of ECG signal" --data_path "path to train dataset" --load_path "path to save checkpoint" 
```

  <li>With Peak-based Error</li>

```
python test.py --dims "dimension of ECG signal" --data_path "path to train dataset" --load_path "path to save checkpoint" --mask_loss
```

</ul>

## Citation
```
@article{tsrnet,
      title={TSRNet: Simple Framework for Real-time ECG Anomaly Detection with Multimodal Time and Spectrogram Restoration Network}, 
      author={Nhat-Tan Bui and Dinh-Hieu Hoang and Thinh Phan and Minh-Triet Tran and Brijesh Patel and Donald Adjeroh and Ngan Le},
      journal={arXiv:2312.10187},
      year={2023}
}
```

## Acknowledgment
A part of this code is adapted from these previous work: [Jiang et al.](https://github.com/MediaBrain-SJTU/ECGAD)

## FAQ
If you have any questions, please feel free to create an issue on this repository or contact us at <tanb@uark.edu> / <hieu.hoang2020@ict.jvn.edu.vn>.
