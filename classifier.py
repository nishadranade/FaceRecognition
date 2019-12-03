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

model = (torch.load('./model1.pt'))


# l = []
# garb = torch.rand(1, 1, 112, 92)
# garb = garb.view(garb.size(0), -1)
# garb = model(garb, l)
# pic = garb.cpu().data
# pic = pic.reshape(112, 92)

def load_dataset(t):
    data_path = 'orlFaces/' + str(t) + '/'
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=320,
        num_workers=0,
        shuffle=False
    )
    return train_loader

data_loader = load_dataset('train')

# get the encodings of all the faces
encodings = []
targets = None

for img, labels in data_loader:
    img = img[:, 0]
    img = img.reshape(320, 1, 112, 92)
    img = img.view(img.size(0), -1)
    img = img.cpu()
    # ============= forward ============
    output = model(img, encodings)
    targets = labels

print(encodings[0].size())
# encodings = encodings[0]

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(112*92, 1500),
            nn.ReLU(),
            nn.Linear(1500, 900),
            nn.ReLU(),
            nn.Linear(900, 500),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 40)
        )  
    def forward(self, x):
        x = self.net(x)
        return x

classifier = classifier().cpu()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    classifier.parameters(), lr = 4e-4, weight_decay = 0)

# print(targets)

# sys.exit()

for e in range(10):
    for img, target in data_loader:
        img = img[:, 0]
        # img = img.reshape(320, 1500)
        img = img.view(img.size(0), -1)
        img = img.cpu()
        # ============= forward ============
        output = classifier(img)
        loss = criterion(output, target)
        print(loss)
        # ============= backward ===========
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        print('*************')
    print('Epoch number: ' + str(e))

test_data = load_dataset('test')

trgt = None
for img, t in test_data:
    img = img[:, 0]
    img = img.view(img.size(0), -1)
    img = img.cpu()
    # ================ forward =============
    output = classifier(img)
    print(output.size())
    _, pred = output.max(1)
    print(pred.size())
    trgt = t

count = 0
for i in range(80):
    if pred[i] == trgt[i]:
        count += 1
    
print(count/80)
