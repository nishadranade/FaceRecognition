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
    data_path = 'orlFaces/'
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=40,
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


# print(type(labels))

images = images[:,0]

print(images.size())

images = images.reshape(40, 1, 112, 92)

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
            nn.Linear(112*92, 2000),
            nn.ReLU(True),
            #nn.BatchNorm1d(2000),
            nn.Linear(2000, 1000),
            nn.ReLU(True),
            #nn.BatchNorm1d(1000),
            nn.Linear(1000, 500),
            nn.ReLU(True), 
            #nn.BatchNorm1d(500),
            nn.Linear(500, 100),
            nn.ReLU(True), 
            nn.Linear(100, 32))
        self.decoder = nn.Sequential(
            nn.Linear(32, 100),
            nn.ReLU(True),
            nn.Linear(100, 500),
            nn.ReLU(True),
            nn.Linear(500, 1000),
            #nn.BatchNorm1d(1000),
            nn.ReLU(True),
            nn.Linear(1000, 2000),
            #nn.BatchNorm1d(2000),
            nn.ReLU(True),
            nn.Linear(2000, 112*92),
            nn.Tanh())
            # nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh()
    def forward(self, x, store):
        x = self.encoder(x)
        store.append(x)
        x = self.decoder(x)
        return x

# img = images[100].numpy()
# print(images[100].size())
# print(img.shape)
# imgplot = plt.imshow(img, cmap='Greys_r')
# plt.show()

# sys.exit()

model = autoencoder().cpu()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr = 1e-3, weight_decay=1e-10)


# store = []
for e in range(5):
    i = 0
    losses = []
    store = []
    for img in images:
        img = img.view(img.size(0), -1)
        # if i == 0:
        #     print('img size while training:'+ str(img.size()))
        img = Variable(img).cpu()
        # ===== forward ==========
        output = model(img, store)
        loss = criterion(output, img)
        losses.append(loss.item())
        # print('loss =' + str(loss.item()))
        # ===== backward =========
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
        # print(i)
        pic =output.cpu().data
        # print("output size= " + str(pic.size()))
        pic = pic.reshape(112, 92)
        save_image(pic, './resultsOrl/pre/decode' +str(i) + '.png')   
    losses = np.array(losses)
    print('********************')
    print('epoch number =' + str(e))
    print('avg loss is = ' + str(np.mean(losses)))    

i = 0
#torch.save(model.state_dict(), './trial_autoenc.pth')

#print(store)

a04 = store[4].detach().numpy()
a05 = store[5].detach().numpy()

a14 = store[14].detach().numpy()
a15 = store[15].detach().numpy()

a24 = store[24].detach().numpy()
a25 = store[25].detach().numpy()

a34 = store[34].detach().numpy()
a35 = store[35].detach().numpy()

print('******* differences between same faces *******')
print(str(np.linalg.norm(a04 - a05)))
print(str(np.linalg.norm(a14 - a15)))
print(str(np.linalg.norm(a24 - a25)))
print(str(np.linalg.norm(a34 - a35)))

print('***** differences between different faces ******')
print(str(np.linalg.norm(a04 - a15)))
print(str(np.linalg.norm(a04 - a14)))
print(str(np.linalg.norm(a04 - a25)))
print(str(np.linalg.norm(a04 - a24)))

store = []

for img in images:
    img = img.view(img.size(0), -1)
    img = Variable(img).cpu()
    # ==== forward ========
    output = model(img, store)
    encoding = store[-1]
    pic2 = encoding.detach().numpy()
    pic2 = pic2.reshape(8, 4)
    save_image(pic2, './resultsOrl/encodings/encode' +str(i) + '.png')
    loss = criterion(output, img)
    print('loss post = ' + str(loss.item()))
    i +=1 
    pic =output.cpu().data
    # print("output size= " + str(pic.size()))
    pic = pic.reshape(112, 92)
    save_image(pic, './resultsOrl/post/decode' +str(i) + '.png')   

