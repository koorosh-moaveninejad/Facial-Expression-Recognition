# Relative Uncertainty Learning for Facial Expression Recognition
In facial expression recognition (FER), the uncertainties introduced by inherent noises like ambiguous facial expressions and inconsistent labels raise concerns about the credibility of recognition results. To quantify these uncertainties and achieve good performance under noisy data, we regard uncertainty as a relative concept and propose an innovative uncertainty learning method called Relative Uncertainty Learning (RUL). Rather than assuming Gaussian uncertainty distributions for all datasets, RUL builds an extra branch to learn uncertainty from the relative difficulty of samples by feature mixup. Specifically, we use uncertainties as weights to mix facial features and design an add-up loss to encourage uncertainty learning. It is easy to implement and adds little or no extra computation overhead. Extensive experiments show that RUL outperforms state-of-the-art FER uncertainty learning methods in both real-world and synthetic noisy FER datasets. Besides, RUL also works well on other datasets such as CIFAR and Tiny ImageNet.



## Train

**Torch** 

We train RUL with Torch 1.8.0 and torchvision 0.9.0.

**Dataset**

Download [RAF-DB](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset), put it into the dataset folder, and make sure that it has the same structure as bellow:
```key
# Facial Expression Recognition with Relative Uncertainty Learning (RUL)

This repository contains my adapted implementation of **Relative Uncertainty Learning (RUL)** for **Facial Expression Recognition (FER)**.

The original RUL method was introduced in:

**Relative Uncertainty Learning for Facial Expression Recognition**  
NeurIPS 2021  
Authors: Yuhang Zhang, Chengrui Wang, Weihong Deng

In this version, I modified the original codebase to work with my own dataset format and to run more reliably on different devices.

---

## Overview

This project is based on the RUL framework for facial expression recognition, with the following practical modifications:

- support for a **custom dataset structure**
- support for **CSV label files**
- support for **separate train/test label files**
- adapted code for **CPU / Apple Silicon MPS / CUDA**
- improved training compatibility for local environments

---

## Dataset Structure

This implementation uses the following dataset format:

```text
DATASET/
├── train/
│   ├── 1/                     # Surprise
│   ├── 2/                     # Fear
│   ├── 3/                     # Disgust
│   ├── 4/                     # Happiness
│   ├── 5/                     # Sadness
│   ├── 6/                     # Anger
│   └── 7/                     # Neutral
├── test/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   ├── 6/
│   └── 7/
├── train_labels.csv
└── test_labels.csv


**Pretrained backbone model**

Download the pretrained ResNet18 from [this](https://drive.google.com/file/d/1EEx7qVCums-TM5fiblepgY70MDqIxbVz/view?usp=sharing) github repository, and then put it into the pretrained_model directory. We thank the authors for providing their pretrained ResNet model.


**Train the RUL model**

```key
cd src
python main.py \
  --raf_path '../../DATASET' \
  --train_label_path '../../DATASET/train_labels.csv' \
  --test_label_path '../../DATASET/test_labels.csv' \
  --pretrained_backbone_path '../pretrained_model/resnet18_msceleb.pth'
```





