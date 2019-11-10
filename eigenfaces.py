from PIL import Image
import numpy as np 
import glob 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
print(org_dataset.shape)
num_components = len(org_dataset)

def normalize(org_dataset):
    mean_vector = np.mean(org_dataset, axis=0)
    dataset = org_dataset - mean_vector
    return dataset, mean_vector

dataset, mean_vector = normalize(org_dataset)

def calculate_eigenvector(dataset):
    cov_mat = np.dot(dataset, dataset.T)
    eig_values, eigen_vectors = np.linalg.eig(cov_mat)
    eig_vectors = np.dot(dataset.T, eigen_vectors)
    for i in range(eig_vectors.shape[1]):
        eig_vectors[:, i] /= np.linalg.norm(eig_vectors[:, i])
    return eig_values.astype(float), eig_vectors.astype(float) 

eig_values, eig_vectors = calculate_eigenvector(dataset)

def pca(eig_values, eig_vectors, k):
    k_eig_val = eig_values.argsort()[-k:][::-1]
    eigen_faces = []

    for i in k_eig_val:
        eigen_faces.append(eig_vectors[:, i])

    eigen_faces = np.array(eigen_faces)
    return eigen_faces

eigen_faces = pca(eig_values, eig_vectors, num_components)

# print(type(eigen_faces))
print(eigen_faces.shape)

def reconstruct_faces(eigen_faces, mean_vector):
    org_dim_eig_faces = []

    for i in range(eigen_faces.shape[0]):
        org_dim_eig_faces.append(eigen_faces[i].reshape(243, 320))
    
    org_dim_eig_faces = np.array(org_dim_eig_faces)
    return org_dim_eig_faces

orig_dim_eig_faces = reconstruct_faces(eigen_faces, mean_vector)

# %matplotlib inline
# plt.plot(eig_values[:10])
# plt.show()
# plt.clf()

def findK(eig_values):
    total_energy = np.trapz(eig_values, dx=1)
    print("Total Energy:", total_energy)

    k = 0
    while True:
        k_energy = np.trapz(eig_values[:k], dx=1)
        print("Energy captured by", k, "components:", k_energy)
        if k_energy >= 0.6*total_energy:
            break
        k += 1

    print("Number of components to capture 60% energy:", k)

# findK(eig_values)

def plot_top10(orig_dim_eig_faces):
    gs = gridspec.GridSpec(2, 5, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)

    i = 0
    for g in gs:
        ax = plt.subplot(g)
        ax.imshow(orig_dim_eig_faces[i], cmap=plt.get_cmap('gray'))
        ax.set_xticks([])
        ax.set_yticks([])
        i += 1
    
    plt.show()
    plt.clf()

plot_top10(orig_dim_eig_faces)