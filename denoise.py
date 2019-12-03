from __future__ import print_function, division
import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils, datasets
from torchvision.utils import save_image
from torch import nn

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(112*92, 7000, bias=True),
            nn.ReLU(True),
            nn.Linear(7000, 4000, bias=True),
            nn.ReLU(True),
            nn.Linear(4000, 2500, bias=True),
            nn.ReLU(True), 
            nn.Linear(2500, 1500, bias=True))
            # nn.ReLU(True)) 
            # nn.Linear(200, 50, bias=False))
        self.decoder = nn.Sequential(
            # nn.Linear(50, 200, bias=False),
            # nn.ReLU(True),
            nn.Linear(1500, 2500, bias=True),
            nn.ReLU(True),
            nn.Linear(2500, 4000, bias=True),
            nn.ReLU(True),
            nn.Linear(4000, 7000, bias=True),
            nn.ReLU(True),
            nn.Linear(7000, 112*92, bias=True)
            )
            # nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh()
    def forward(self, x, store):
        x = self.encoder(x)
        store.append(x)
        x = self.decoder(x)
        return x

model = autoencoder().cpu()

# load pretrained model
model = (torch.load('./modelOld.pt'))

# get a couple testing samples, with shuffle true
def load_dataset(t):
    data_path = 'orlFaces/' + str(t) + '/'
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=0,
        shuffle=True
    )
    return train_loader

data_loader = load_dataset('train')

output = None
for img, _ in data_loader:
    img = img[:, 0]
    img = img.reshape(2, 1, 112, 92)
    img = img.view(img.size(0), -1)
    print(img.size())
    # add noise to image p = 0.2
    noise = torch.rand(2, 10304)
    noise = (noise > 0.25).int()
    img = img * noise
    # ======= forward =========
    output = model(img, [])
    break

pic = output.data
for i, p in enumerate(pic):
    p = p.reshape(112, 92)
    save_image(p, './denoise/d' + str(i) + '.png')

