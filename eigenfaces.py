from PIL import Image
import numpy as np 
import glob 

def build_dataset():
    org_dataset = []

    for i in range(1, 16):
        filelist = glob.glob('./eigenData/train/subject'+str(i).zfill(2)+"*.png")
        for fname in filelist:
            img = np.array(Image.open(fname))
            img = img.reshape(img.shape[0] * img.shape[1])
            org_dataset.append(img)

    org_dataset = np.array(org_dataset)
    return org_dataset

org_dataset = build_dataset()
print(len(org_dataset))