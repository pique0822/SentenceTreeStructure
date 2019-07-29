import torch.nn as nn
import torch

from Gated_GRU import GatedGRU
from datasets.arithmetic.arithmetic_dataset import Dataset

import numpy as np
import random

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


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
#0 - sign, 1 - sign/hundreds, 2 - sign/tens, 3 - ones, 4-plus,
#5 - sign, 6 - sign/hundreds, 7 - sign/tens, 8 - ones, 9-equals
temporal_decoding = {0:[],1:[],2:[],3:[],4:[],
                     5:[],6:[],7:[],8:[],9:[]}
temporal_labels_first_num = {0:[],1:[],2:[],3:[],4:[],
                     5:[],6:[],7:[],8:[],9:[]}
temporal_labels_second_num = {0:[],1:[],2:[],3:[],4:[],
                     5:[],6:[],7:[],8:[],9:[]}

temporal_labels_running_sum = {0:[],1:[],2:[],3:[],4:[],
                     5:[],6:[],7:[],8:[],9:[]}

for idx in range(dataset.testing_size()):
    running_sum = []

    input, label, line = dataset.testing_item(idx)

    addition_index = line.index('+')
    equals_index = line.index('=')

    first_num = line[:addition_index]
    second_num = line[addition_index+1:equals_index]

    true_first_num = int(first_num)
    true_second_num = int(second_num)

    x = torch.Tensor(input).reshape(1, -1, dataset.vector_size)

    hidden = model.init_hidden()

    decoded, hidden, (update_gates, reset_gates, hidden_states) = model(x,hidden)

    prediction = decoded[len(decoded)-1].item()

    previous_sum = 0
    for char_idx in range(len(line)):
        sub_line = line[:char_idx+1]

        try:
            parsum = eval(sub_line)
            previous_sum = parsum
        except:
            pass

        running_sum.append(previous_sum)

    negative_first = 0
    if first_num[0] == '-':
        negative_first = 1
        temporal_decoding[0].append(hidden_states[0].detach().numpy().reshape(-1))
        first_num = first_num[1:]
        temporal_labels_first_num[0].append(true_first_num)
        temporal_labels_second_num[0].append(true_second_num)
        temporal_labels_running_sum[0].append(running_sum[0])


    num_index = 4 - len(first_num)
    for i in range(0,len(first_num)):
        temporal_decoding[num_index + i].append(hidden_states[i + negative_first].detach().numpy().reshape(-1))
        temporal_labels_first_num[num_index + i].append(true_first_num)
        temporal_labels_second_num[num_index + i].append(true_second_num)
        temporal_labels_running_sum[num_index + i].append(running_sum[i + negative_first])

    temporal_decoding[4].append(hidden_states[addition_index].detach().numpy().reshape(-1))
    temporal_labels_first_num[4].append(true_first_num)
    temporal_labels_second_num[4].append(true_second_num)
    temporal_labels_running_sum[4].append(running_sum[addition_index])

    negative_second = 0
    if second_num[0] == '-':
        negative_second = 1
        temporal_decoding[5].append(hidden_states[addition_index+1].detach().numpy().reshape(-1))
        second_num = second_num[1:]
        temporal_labels_first_num[5].append(true_first_num)
        temporal_labels_second_num[5].append(true_second_num)
        temporal_labels_running_sum[5].append(running_sum[addition_index+1])

    num_index = 4 - len(second_num)
    for i in range(0,len(second_num)):
        temporal_decoding[5+num_index + i].append(hidden_states[i+addition_index+1 + negative_second].detach().numpy().reshape(-1))
        temporal_labels_first_num[5+num_index + i].append(true_first_num)
        temporal_labels_second_num[5+num_index + i].append(true_second_num)
        temporal_labels_running_sum[5+num_index + i].append(running_sum[i+addition_index+1 + negative_second])

    temporal_decoding[9].append(hidden_states[equals_index].detach().numpy().reshape(-1))
    temporal_labels_first_num[9].append(true_first_num)
    temporal_labels_second_num[9].append(true_second_num)
    temporal_labels_running_sum[9].append(running_sum[equals_index])

pca_num_1 = PCA(n_components=100)
pca_num_2 = PCA(n_components=100)
pca_all = PCA(n_components=100)

for key in temporal_decoding:
    print(len(temporal_decoding[key]),len(temporal_labels_first_num[key]),len(temporal_labels_second_num[key]),len(temporal_labels_running_sum[key]))
# number_1
individual_pca_temporal_num_1 = {}

temp_array = np.array(temporal_decoding[0])
for key in range(1,4):
    temp_array = np.vstack((temp_array,np.array(temporal_decoding[key])))

pca_num_1.fit(temp_array)
for key in temporal_decoding.keys():
    if len(temporal_decoding[key]) > 0:
        individual_pca_temporal_num_1[key] = pca_num_1.transform(np.array(temporal_decoding[key]))

reg = LogisticRegression()
reg.fit(individual_pca_temporal_num_1[3],temporal_labels_first_num[3])

print('Number 1 on Number 1')
for key in individual_pca_temporal_num_1.keys():
    print(reg.score(individual_pca_temporal_num_1[key],temporal_labels_first_num[key]))
print('Number 1 on Number 2')
for key in individual_pca_temporal_num_1.keys():
    print(reg.score(individual_pca_temporal_num_1[key],temporal_labels_second_num[key]))




# number_2
individual_pca_temporal_num_2 = {}
temp_array = np.array(temporal_decoding[0])
for key in range(5,9):
    temp_array = np.vstack((temp_array,np.array(temporal_decoding[key])))

pca_num_2.fit(temp_array)
for key in temporal_decoding.keys():
    if len(temporal_decoding[key]) > 0:
        individual_pca_temporal_num_2[key] = pca_num_2.transform(np.array(temporal_decoding[key]))

reg = LogisticRegression()
reg.fit(individual_pca_temporal_num_2[8],temporal_labels_second_num[8])

print('Number 2 on Number 2')
for key in individual_pca_temporal_num_2.keys():
    print(reg.score(individual_pca_temporal_num_2[key],temporal_labels_second_num[key]))
print('Number 2 on Number 1')
for key in individual_pca_temporal_num_2.keys():
    print(reg.score(individual_pca_temporal_num_2[key],temporal_labels_first_num[key]))


reg = LogisticRegression()
reg.fit(individual_pca_temporal_num_2[8],temporal_labels_running_sum[8])
print('Running Sum Number 2 on Number 2')
for key in individual_pca_temporal_num_2.keys():
    print(reg.score(individual_pca_temporal_num_2[key],temporal_labels_running_sum[key]))

print('Running Sum Number 2 on Number 1')
for key in individual_pca_temporal_num_1.keys():
    print(reg.score(individual_pca_temporal_num_1[key],temporal_labels_running_sum[key]))

# import pdb; pdb.set_trace()


# number_1
individual_pca_temporal_all = {}

temp_array = np.array(temporal_decoding[0])
for key in temporal_decoding.keys():
    temp_array = np.vstack((temp_array,np.array(temporal_decoding[key])))

pca_all.fit(temp_array)
for key in temporal_decoding.keys():
    if len(temporal_decoding[key]) > 0:
        individual_pca_temporal_all[key] = pca_all.transform(np.array(temporal_decoding[key]))

temp_array = np.array(individual_pca_temporal_all[0])
temp_labels = list(temporal_labels_running_sum[0])
for key in range(1,4):
    temp_array = np.vstack((temp_array,np.array(individual_pca_temporal_all[key])))
    temp_labels.extend(temporal_labels_running_sum[key])

reg = LogisticRegression()
reg.fit(temp_array, temp_labels)

print('Running Sum Number 1 on All')
for key in individual_pca_temporal_all.keys():
    print(reg.score(individual_pca_temporal_all[key],temporal_labels_running_sum[key]))


temp_array = np.array(individual_pca_temporal_all[5])
temp_labels = list(temporal_labels_running_sum[5])
for key in range(6,9):
    temp_array = np.vstack((temp_array,np.array(individual_pca_temporal_all[key])))
    temp_labels.extend(temporal_labels_running_sum[key])

reg = LogisticRegression()
reg.fit(temp_array, temp_labels)

print('Running Sum Number 2 on All')
for key in individual_pca_temporal_all.keys():
    print(reg.score(individual_pca_temporal_all[key],temporal_labels_running_sum[key]))
