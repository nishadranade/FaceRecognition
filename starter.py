from __future__ import print_function, division
import os
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

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

def load_dataset():
    data_path = 'yaleFaces/data/train/'
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=135,
        num_workers=0,
        shuffle=False
    )
    return train_loader

data_loader = load_dataset()
# print(type(data_loader[1])


images = None
labels = None

for i, l in data_loader:
    images = i
    labels = l

print(images.size())
# print(type(labels))

images = images[:,0]

images = images.reshape(135, 1, 243, 320)

print(images.size())
print(labels.size())
# print(images[0][0][23])

# code for autoencoder

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(243*320, 5000),
            nn.ReLU(True),
            nn.Linear(5000, 1000),
            nn.ReLU(True), 
            nn.Linear(1000, 100))
            # nn.ReLU(True), 
            # nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(100, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 5000),
            nn.ReLU(True),
            nn.Linear(5000, 243*320))
            # nn.Tanh())
            # nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh()
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# img = images[100].numpy()
# print(images[100].size())
# print(img.shape)
# imgplot = plt.imshow(img, cmap='Greys_r')
# plt.show()

model = autoencoder().cpu()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr = 1e-3, weight_decay=1e-5)

i = 0

for img in images:
    img = img.view(img.size(0), -1)
    img = Variable(img).cpu()
    # ===== forward ==========
    output = model(img)
    loss = criterion(output, img)
    # ===== backward =========
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    i += 1
    # pic =output.cpu().data
    # print("output size= " + str(pic.size()))
    # pic = pic.reshape(243, 320)
    # save_image(pic, './decoded/decode' +str(i) + '.png')   
    
i = 0
# torch.save(model.state_dict(), './trial_autoenc.pth')
for img in images:
    img = img.view(img.size(0), -1)
    img = Variable(img).cpu()
    # ==== forward ========
    output = model(img)
    i +=1 
    pic =output.cpu().data
    print("output size= " + str(pic.size()))
    pic = pic.reshape(243, 320)
    save_image(pic, './results5000/decode' +str(i) + '.png')   