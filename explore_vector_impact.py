import torch.nn as nn
import torch

from Gated_GRU import GatedGRU
from datasets.arithmetic.arithmetic_dataset import Dataset

import numpy as np
import random

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier

from textwrap import wrap

from tqdm import tqdm

dataset_training = 'datasets/arithmetic/1e2/training.txt'
dataset_testing = 'datasets/arithmetic/1e2/testing.txt'

dataset = Dataset(dataset_training,dataset_testing)

num_layers = 1
hidden_size = 100
num_epochs = 5
input_size = dataset.vector_size
PATH = 'models/arithmetic_l_'+str(num_layers)+'_h_'+str(hidden_size)+'_ep_'+str(num_epochs)

model = GatedGRU(dataset.vector_size,hidden_size,output_size=1)
model.load_state_dict(torch.load(PATH))
model.eval()

wiu = model.W_iu.detach()
whu = model.W_hu.detach()
bu = model.b_u.detach()

random_samples = 10000

update_gates = []
classes = []

color_dict = {0:'#c56cf0',1:'#ffb8b8',2:'#ff3838',3:'#ff9f1a',4:'#fff200',5:'#3ae374',6:'#67e6dc',7:'#17c0eb',8:'#7158e2',9:'#3d3d3d',10:'#0984e3',11:'#fdcb6e',12:'#7f8c8d'}

for character in tqdm(range(13)):

    input = torch.zeros(1,input_size)
    input[0,character] = 1
    for sample in range(random_samples):
        # sample n random hidden states such that values range from -1 to 1
        # compare the populations of hidden states with different inputs
        random_hidden_state = 2*(torch.rand(1,hidden_size) - 0.5)

        z_t = torch.sigmoid(input @ wiu + random_hidden_state @ whu + bu)

        update_gates.append(z_t.numpy())
        if character is not 9:
            classes.append(0)
        else:
            classes.append(1)


update_gates = np.array(update_gates).reshape(-1,hidden_size)

cls = RidgeClassifier()
cls.fit(update_gates,classes)
print(cls.score(update_gates,classes))


pca = PCA(n_components=3)
pcad_update = pca.fit_transform(update_gates)
print('Var',np.sum(pca.explained_variance_ratio_))

# plt.scatter(pcad_update[:,0],pcad_update[:,1],c=classes)
# plt.show()
