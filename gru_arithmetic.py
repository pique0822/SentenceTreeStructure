import torch.nn as nn
import torch

from Gated_GRU import GatedGRU
from datasets.arithmetic.arithmetic_dataset import Dataset

import numpy as np
import random

import argparse

from tqdm import tqdm

import matplotlib.pyplot as plt

dataset_training = 'datasets/arithmetic/fixed_1e2/training.txt'
dataset_testing = 'datasets/arithmetic/fixed_1e2/testing.txt'

dataset = Dataset(dataset_training,dataset_testing)

parser = argparse.ArgumentParser(description='Training GRU model')
parser.add_argument('--num_epochs', type=int, default='5',
                    help='Number of epochs.')
parser.add_argument('--num_layers', type=int, default='1',
                    help='Number of epochs.')
parser.add_argument('--hidden_size', type=int, default='3',
                    help='Number of epochs.')
parser.add_argument('--use_cuda', type=str, default='False',
                    help='Flag to use cuda or not.')

args = parser.parse_args()

if not torch.cuda.is_available() and args.use_cuda == 'True':
    print('CUDA UNAVAILABLE')
    raise ValueError

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


input_size = dataset.vector_size
PATH = 'arithmetic_1e2_fixed_l_'+str(args.num_layers)+'_h_'+str(args.hidden_size)+'_ep_'+str(args.num_epochs)


model = GatedGRU(input_size,args.hidden_size,output_size=1)

model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

all_loss = []

for epoch in tqdm(range(args.num_epochs)):
    epoch_loss = 0

    for idx in range(dataset.training_size()):
        optimizer.zero_grad()

        hidden = model.init_hidden()

        input, label, line = dataset.training_item(idx)

        x = torch.Tensor(input).reshape(1, -1, dataset.vector_size)
        x.to(device)


        output, hidden, _ = model(x,hidden)

        loss = criterion(output[len(output)-1], torch.Tensor([label]).to(device))

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print('Epoch:',epoch,'Loss:',epoch_loss)
    all_loss.append(epoch_loss)

    dataset.shuffle_order()
    if (epoch+1) % 25 == 0
        SUB_PATH = 'arithmetic_1e2_fixed_l_'+str(args.num_layers)+'_h_'+str(args.hidden_size)+'_ep_'+str(epoch)

        torch.save(model.state_dict(), SUB_PATH)

plt.title('MSELoss Over Epochs')
plt.plot(range(args.num_epochs),all_loss)
plt.xlabel('Epoch')
plt.ylabel('MSELoss')
# plt.show()
plt.savefig(PATH+'_training_loss.png')
plt.close()

torch.save(model.state_dict(), PATH)
