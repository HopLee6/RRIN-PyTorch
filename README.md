# Video Frame Interpolation via Residue Refinement (RRIN)
 [Paper](https://ieeexplore.ieee.org/document/9053987/)

Haopeng Li, Yuan Yuan, [Qi Wang](http://crabwq.github.io/#top)

IEEE Conference on Acoustics, Speech, and Signal Processing, ICASSP 2020



### Table of Contents
1. [Introduction](#introduction)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Pre-trained Model](#Pre-trained-Model)
1. [Testing Demo](#Testing-Demo)
1. [Downloading Results](#downloading-results)



### Introduction
We propose Video Frame **In**terpolation via **R**esidue **R**efinement (**RRIN**) that leverages residue refinement and adaptive weight to synthesize in-between frames. 

Residue refinement is used for optical flow and image generation for higher accuracy and better visual appearance, while the adaptive weight map combines the forward and backward warped frames to reduce the artifacts. 

All sub-modules in our method are implemented by U-Net with various depths.

Experiments on public datasets demonstrate the effectiveness and superiority of our method over the state-of-the-art approaches.



### Requirements and Dependencies
- Python = 3.6.8 in Anaconda3 = 4.7.5
- CUDA = 9.2 & cuDNN = 7.0
- PyTorch = 1.0



### Installation
Download repository:

    $ git clone https://github.com/HopLee6/RRIN.git



### Pre-trained Model

We provide the pre-trained model of "RRIN" at [OneDrive](https://1drv.ms/u/s!AsFdN0iAbWxBjIBWVVsdImS6md0jlA?e=1b14MH), which achieves the same results as reported in the paper. Download the pre-trained model to `/RRIN`.



### Testing Demo

Test the model using frames in `/RRIN/data`:

```
$ python demo.py
```

and get the interpolated frame `/RRIN/data/im_interp.png`.



### Downloading Results
Our RRIN model achieves the state-of-the-art performance on Vimeo90K, and comparable performance on UCF101. Download our interpolated results:

[Vimeo90K](https://1drv.ms/u/s!AsFdN0iAbWxBjIBYTVYPA5-3RPGQmg?e=LJ2Q1F)

[UCF101](https://1drv.ms/u/s!AsFdN0iAbWxBjIBXnNcOEEmElKqsww?e=4s9eeo)



### Contact
[Haopeng Li](mailto:hplee@mail.nwpu.edu.cn)



### License and Citation

The use of this code is RESTRICTED to **non-commercial research and educational purposes**.

```
@INPROCEEDINGS{RRIN, 
author={Haopeng, Li and Yuan, Yuan and Qi, Wang}, 
booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
title={Video Frame Interpolation Via Residue Refinement}, 
year={2020}, 
pages={2613-2617}
}
```


