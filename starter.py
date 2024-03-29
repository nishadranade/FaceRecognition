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

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

def load_dataset():
    data_path = 'yaleFaces/data/train'
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=93,
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

images = images.reshape(33, 1, 243, 320)

print(images.size())
print(labels)
# print(images[0][0][23])
# code for autoencoder

# ============ subtract mean image from all the images ===========
# print('before subtracting mean ' + str(images.size()))

# mean_img = torch.mean(images, axis=0)
# images = images - mean_img

# print('after subtracting mean ' + str(images.size()))

# img = images[5].reshape(243, 320).numpy()
# print(images[5].size())
# print(img.shape)
# imgplot = plt.imshow(img, cmap='Greys_r')
# plt.show()
# sys.exit()


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(243*320, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 500),
            nn.ReLU(True), 
            nn.Linear(500, 100),
            nn.ReLU(True), 
            nn.Linear(100, 32))
        self.decoder = nn.Sequential(
            nn.Linear(32, 100),
            nn.ReLU(True),
            nn.Linear(100, 500),
            nn.ReLU(True),
            nn.Linear(500, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 243*320),
            nn.Tanh())
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
    model.parameters(), lr = 1e-5, weight_decay=1e-5)

i = 0
for e in range(100):
    i = 0
    losses = []
    for img in images:
        img = img.view(img.size(0), -1)
        # if i == 0:
        #     print('img size while training:'+ str(img.size()))
        img = Variable(img).cpu()
        # ===== forward ==========
        output = model(img)
        loss = criterion(output, img)
        losses.append(loss.item())
        # ===== backward =========
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
        # print(i)
        pic =output.cpu().data
        # print("output size= " + str(pic.size()))
        pic = pic.reshape(243, 320)
        save_image(pic, './results10/pre/decode' +str(i) + '.png')   
    losses = np.array(losses)
    print('********************')
    print('epoch number =' + str(e))
    print('avg loss is = ' + str(np.mean(losses)))    

        
i = 0
#torch.save(model.state_dict(), './trial_autoenc.pth')

for img in images:
    img = img.view(img.size(0), -1)
    img = Variable(img).cpu()
    # ==== forward ========
    output = model(img)
    i +=1 
    pic =output.cpu().data
    print("output size= " + str(pic.size()))
    pic = pic.reshape(243, 320)
    save_image(pic, './results10/post/decode' +str(i) + '.png')   