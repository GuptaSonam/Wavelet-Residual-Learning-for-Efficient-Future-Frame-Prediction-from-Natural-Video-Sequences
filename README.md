# WRL-Net: Wavelet-Residual-Learning-for-Efficient-Future-Frame-Prediction-from-Natural-Video-Sequences

Accepted for Oral in 9th International Conference on Pattern Recognition and Machine Intelligence (PReMI'21)

This is an official implementation in PyTorch of WRL-Net. 

![image](https://user-images.githubusercontent.com/8327102/143667515-9d41427d-26f2-4735-90cf-14d10576662c.png)

# Abstract
Video prediction is a challenging task of predicting the future
frames based upon the past frames. Existing deep learning (DL) based
methods either directly hallucinate the pixel values in high-dimensional
video space, resulting in blurry predictions, or decompose the input space
into lower-dimensional intermediate representations requiring additional
assumptions. Moreover, due to the complexity of the task, existing meth-
ods often propose complex networks with high memory and compute
requirements. To address these limitations, we propose a simpler resid-
ual based architecture in wavelet domain for faster and accurate video
prediction. The natural sparsity of wavelet domain makes the learning
task easier for the model. To the best of our knowledge, this is the first
DL-based method that predicts future frames entirely in the wavelet
domain. Our approach takes 2-dimensional Discrete Wavelet Transform
(2D-DWT) sub-bands of video frames as input and learns to infer the
difference between the wavelet coeficients of the adjacent frames (Temporal
Wavelet Residuals). Final prediction is obtained by adding the input
to the predicted residuals followed by application of Inverse Discrete
Wavelet Transform (IDWT). The sparsity of wavelet residuals reduces
the training and inference time. Extensive experimentation demonstrates
that the proposed approach is computationally efficient and still competitive
with the state-of-the-art methods both qualitatively and quantitatively,
on KTH and KITTI datasets.

# Summary
* First DL appraoch that predicts future frames in wavelet domain.
* Predicts residual wavelet coefficients between adjacent frames instead of reconstructing every frame from scratch which leads to faster convergence during training.
* Efficient in both training and inference time wihle achieving performance comparable to SOTA methods.

# Qualitative Performance
![image](https://user-images.githubusercontent.com/8327102/143667685-82b9c0fa-2f6e-49bf-9402-6300f6a0debc.png)

# Dependencies
Python == 3.5 

PyTorch == 1.1.0

CUDA == 9.0

CUDNN == 7.5.1

PyWavelets == 1.0.0

# Installation

- Install python 3.5 and other dependencies as mentioned above.
- Download [KTH](https://www.csc.kth.se/cvap/actions/) Human Action Dataset and perform preprocessing as specifed in the paper.



# Training
To Train the network, use the following bas script after setting appropriate parameters:
```bash
./predrnn_kth_IDWT_1branch_res.sh
```


# Citation
If you find this work useful for your research, please use the following BibTeX entry to cite our paper.
```
@inproceedings{gupta2021g3an++,
  title={Wavelet Residual Learning for Efficient Future Frame Prediction from Natural Video Sequences,
  author={Gupta, Sonam and Das, Sukhendu},
  booktitle={Proceedings of the 9th International Conference on Pattern Recognition and Machine Learning},
  year={2021}
}

```





