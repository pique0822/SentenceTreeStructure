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
from textwrap import wrap

from tqdm import tqdm

dataset_training = 'datasets/arithmetic/fixed_L4_1e3/training.txt'
dataset_testing = 'datasets/arithmetic/fixed_L4_1e3/testing.txt'

dataset = Dataset(dataset_training,dataset_testing)

hidden_sizes = [100]

losses = []
for hidden in hidden_sizes:
    print('Testing:',hidden)
    total_loss = 0

    num_layers = 1
    hidden_size = hidden
    num_epochs = 249
    input_size = dataset.vector_size
    PATH = 'models/arithmetic_L4_1e3_fixed_l_'+str(num_layers)+'_h_'+str(hidden_size)+'_ep_'+str(num_epochs)

    model = GatedGRU(dataset.vector_size,hidden_size,output_size=1)
    model.load_state_dict(torch.load(PATH))
    model.eval()


    for idx in tqdm(range(dataset.training_size())):
        input, label, line = dataset.training_item(idx)

        addition_index = line.index('+')

        equals_index = line.index('=')

        x = torch.Tensor(input).reshape(1, -1, dataset.vector_size)

        hidden = model.init_hidden()

        decoded, hidden, (update_gates, reset_gates, hidden_states) = model(x,hidden)

        prediction = decoded[len(decoded)-1].item()

        total_loss += (prediction - label)**2


    total_loss = total_loss / dataset.testing_size()

    losses.append(total_loss)
print(losses)
plt.title('Test MSE Over Hidden Size')
plt.ylabel('MSE')
plt.plot(range(len(losses)),losses)
plt.xticks(range(len(losses)),hidden_sizes)
plt.xlabel('Hidden Size')
plt.show()
