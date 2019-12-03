import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils, datasets
from torchvision.utils import save_image
from torch import nn

def load_dataset():
    data_path = 'umist/'
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )
    return train_loader

data_loader = load_dataset()

a = None
b = None

for img, _ in data_loader:
    img = img[:, 0]
    print(img.size())
    img = img.reshape(2712, 2232)
    a = img[:112, 92*3 + 3 : 92*4 + 3]
    b = img[:112, :92]

print(a.size())

save_image(a, './umist/03.png')
save_image(b, './umist/00.png')

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
model = (torch.load('./model1.pt'))

a = a.reshape(1,1,112,92)
img = a
# img = img.view(img.size(0), -1)
img = img.reshape(112*92,)
output = model(img, [])

pic = output.data

print(pic.size())

pic = pic.reshape(112, 92)
save_image(pic, './umist/dec.png')

# for i, p in enumerate(pic):
#     print(p.size())
#     p = p.reshape(112, 92)
    