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
    data_path = 'orlFaces/train/'
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


# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(112*92, 1000, bias=True)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(1000, 112*92),
#             nn.Tanh()
#         )
#     def forward(self, x, store):
#         x = self.encoder(x)
#         store.append(x)
#         x = self.decoder(x)
#         return x

# img = images[100].numpy()
# print(images[100].size())
# print(img.shape)
# imgplot = plt.imshow(img, cmap='Greys_r')
# plt.show()

# sys.exit()

model = autoencoder().cpu()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr = 2e-5, weight_decay=0)


losses = []

picE = None

for e in range(2000):
    store = []
    lossE = 0
    for img, _ in data_loader:              #img is now a batch
        img = img[:,0]
        print(img.size())
        img = img.reshape(40, 1, 112, 92)
        print(img.size())
        img = img.view(img.size(0), -1)
        print(img.size())
        img = img.cpu()
        # ============= forward ============
        output = model(img, store)
        loss = criterion(output, img)
        print(loss)
        lossE = loss
        # ============ backward ============
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #i += 1
        # print(i)
        pic =output.cpu().data
        print(pic.size())
        picE = pic
        # print("output size= " + str(pic.size()))  
        print('********************')
    losses.append(lossE)
    print('Epoch number: ' + str(e))


for i, p in enumerate(picE):
    p = p.reshape(112, 92)
    save_image(p, './resultsOrl/pre/decode' +str(i) + '.png') 


# i = 0
#torch.save(model.state_dict(), './trial_autoenc.pth')

# ********************** new block ***************

# for img, _ in data_loader:
#     img = img[:, 0]
#     for i, p in enumerate(img):
#         p = p.reshape(112, 92)
#         save_image(p, './resultsOrl/orig/orig' +str(i) + '.png')  
#     break


garb = torch.rand(1,1,112,92)
garb = garb.view(garb.size(0), -1)
garb = model(garb, [])
pic = garb.cpu().data
pic = pic.reshape(112, 92)
#save_image(pic, './resultsOrl/orig/junk.png') 

plt.plot(losses[900:])
plt.savefig('./lossPlot.png')


for img, _ in data_loader:              #img is now a batch
    img = img[:,0]
    img = img.reshape(40, 1, 112, 92)
    img = img.view(img.size(0), -1)
    img = img.cpu()
    # ============= forward ============
    output = model(img, store)
    # ============ backward ============
    pic =output.cpu().data
    for i, p in enumerate(pic):
        p = p.reshape(112, 92)
        save_image(p, './resultsOrl/post/decode' +str(i) + '.png')   
    break


torch.save(model, './model1.pt')

sys.exit()




# ********************************** irrelevant, wont run anyway ****************


# a04 = store[4].detach().numpy()
# a05 = store[5].detach().numpy()

# a14 = store[14].detach().numpy()
# a15 = store[15].detach().numpy()

# a24 = store[24].detach().numpy()
# a25 = store[25].detach().numpy()

# a34 = store[34].detach().numpy()
# a35 = store[35].detach().numpy()

# print('******* differences between same faces *******')
# print(str(np.linalg.norm(a04 - a05)))
# print(str(np.linalg.norm(a14 - a15)))
# print(str(np.linalg.norm(a24 - a25)))
# print(str(np.linalg.norm(a34 - a35)))

# print('***** differences between different faces ******')
# print(str(np.linalg.norm(a04 - a15)))
# print(str(np.linalg.norm(a04 - a14)))
# print(str(np.linalg.norm(a04 - a25)))
# print(str(np.linalg.norm(a04 - a24)))

# store = []

# for img in images:
#     save_image(img, './resultsOrl/orig/orig' + str(i)+ '.png')
#     img = img.view(img.size(0), -1)
#     img = Variable(img).cpu()
#     # ==== forward ========
#     output = model(img, store)
#     encoding = store[-1]
#     pic2 = encoding.cpu().data
#     pic2 = pic2.reshape(10, 5)
#     save_image(pic2, './resultsOrl/encodings/encode' +str(i) + '.png')
#     loss = criterion(output, img)
#     print('loss post = ' + str(loss.item()))
#     i +=1 
#     pic =output.cpu().data
#     # print("output size= " + str(pic.size()))
#     pic = pic.reshape(112, 92)
#     save_image(pic, './resultsOrl/post/decode' +str(i) + '.png')   

# garb = torch.zeros(1,1,112,92)
# print(garb.size())
# l = []

# garb = garb.view(garb.size(0), -1)
# garb = Variable(garb).cpu()

# garb = model(garb, l)
# pic = garb.cpu().data
# pic = pic.reshape(112, 92)
# save_image(pic, './resultsOrl/orig/junk.png') 
