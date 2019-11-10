from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

def load_dataset():
    data_path = 'yaleFaces/data/test/'
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader

data_loader = load_dataset()
# print(len(data_loader))
for image, label in data_loader:
    print(image.size())
    print(len(label))
    print(image[0][2][0][0])
