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
from sklearn.linear_model import LinearRegression

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


dataset_training = 'datasets/arithmetic/fixed_1e2/training.txt'
dataset_testing = 'datasets/arithmetic/fixed_1e2/testing.txt'

dataset = Dataset(dataset_training,dataset_testing)

num_layers = 1
hidden_size = 100
num_epochs = 449
input_size = dataset.vector_size
PATH = 'models/arithmetic_1e2_fixed_l_'+str(num_layers)+'_h_'+str(hidden_size)+'_ep_'+str(num_epochs)

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
temporal_labels_partial_num = {0:[],1:[],2:[],3:[],4:[],
                     5:[],6:[],7:[],8:[],9:[]}

for idx in range(dataset.testing_size()):
    running_sum = []
    partial_num = []

    input, label, line = dataset.testing_item(idx)

    addition_index = 4
    equals_index = 9

    first_num = line[:addition_index]
    second_num = line[addition_index+1:equals_index]

    true_first_num = int(first_num)
    true_second_num = int(second_num)

    x = torch.Tensor(input).reshape(1, -1, dataset.vector_size)

    hidden = model.init_hidden()

    decoded, hidden, (update_gates, reset_gates, hidden_states) = model(x,hidden)

    prediction = decoded[len(decoded)-1].item()

    for char_idx in range(len(line)):
        sub_line = line[:char_idx+1]
        num_one = 0
        num_two = 0
        if len(sub_line) <= 5:
            try:
                num_one = int(line[:min(char_idx+1,4)])
            except:
                num_one = 0
        if len(sub_line) > 5:
            num_one = int(line[:4])
            try:
                num_two = int(line[5:min(char_idx+1,9)])
            except:
                num_two = 0

        running_sum.append(num_one + num_two)

    previous_sum = 0
    for char_idx in range(len(line)):
        if char_idx > addition_index:
            sub_line = line[addition_index+1:char_idx+1]
            previous_sum = 0
        else:
            sub_line = line[:char_idx+1]

        try:
            parsum = eval(sub_line)
            previous_sum = parsum
        except:
            pass


        partial_num.append(previous_sum)

    for char_idx in range(len(line)):
        temporal_decoding[char_idx].append(hidden_states[char_idx].detach().numpy().reshape(-1))
        temporal_labels_first_num[char_idx].append(true_first_num)
        temporal_labels_second_num[char_idx].append(true_second_num)
        temporal_labels_running_sum[char_idx].append(running_sum[char_idx])
        temporal_labels_partial_num[char_idx].append(partial_num[char_idx])


# UnPCA'd Analysis
# FIRST NUM
first_num_scores = np.zeros((len(temporal_decoding.keys()),len(temporal_decoding.keys())))

for col, training_key in enumerate(temporal_decoding.keys()):
    training_data = np.array(temporal_decoding[training_key])
    training_labels = np.array(temporal_labels_first_num[training_key])

    reg = LinearRegression()
    reg.fit(training_data, training_labels)

    for row, testing_key in enumerate(temporal_decoding.keys()):
        testing_data = np.array(temporal_decoding[testing_key])
        testing_labels = np.array(temporal_labels_first_num[testing_key])

        score = reg.score(testing_data, testing_labels)

        first_num_scores[row,col] = score

plt.title('Confusion Matrix of First Number Memory')
plt.xlabel('Trained On')
plt.ylabel('Testing On')
plt.xticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.yticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.imshow(first_num_scores, origin='lower', vmin=0,vmax=1)
plt.colorbar()
plt.show()


second_num_scores = np.zeros((len(temporal_decoding.keys()),len(temporal_decoding.keys())))

for col, training_key in enumerate(temporal_decoding.keys()):
    training_data = np.array(temporal_decoding[training_key])
    training_labels = np.array(temporal_labels_second_num[training_key])

    reg = LinearRegression()
    reg.fit(training_data, training_labels)

    for row, testing_key in enumerate(temporal_decoding.keys()):
        testing_data = np.array(temporal_decoding[testing_key])
        testing_labels = np.array(temporal_labels_second_num[testing_key])

        score = reg.score(testing_data, testing_labels)

        second_num_scores[row,col] = score

plt.title('Confusion Matrix of Second Number Memory')
plt.xlabel('Trained On')
plt.ylabel('Testing On')
plt.xticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.yticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.imshow(second_num_scores, origin='lower', vmin=0,vmax=1)
plt.colorbar()
plt.show()



running_sum_scores = np.zeros((len(temporal_decoding.keys()),len(temporal_decoding.keys())))

for col, training_key in enumerate(temporal_decoding.keys()):
    training_data = np.array(temporal_decoding[training_key])
    training_labels = np.array(temporal_labels_running_sum[training_key])
    try:
        reg = LinearRegression()
        reg.fit(training_data, training_labels)

        for row, testing_key in enumerate(temporal_decoding.keys()):
            testing_data = np.array(temporal_decoding[testing_key])
            testing_labels = np.array(temporal_labels_running_sum[testing_key])

            score = reg.score(testing_data, testing_labels)

            running_sum_scores[row,col] = score
    except:
        for row, testing_key in enumerate(temporal_decoding.keys()):
            running_sum_scores[row,col] = 0

plt.title('Confusion Matrix of Running Sum Memory')
plt.xlabel('Trained On')
plt.ylabel('Testing On')
plt.xticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.yticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.imshow(running_sum_scores, origin='lower', vmin=0,vmax=1)
plt.colorbar()
plt.show()



partial_num_score = np.zeros((len(temporal_decoding.keys()),len(temporal_decoding.keys())))

for col, training_key in enumerate(temporal_decoding.keys()):
    training_data = np.array(temporal_decoding[training_key])
    training_labels = np.array(temporal_labels_partial_num[training_key])
    try:
        reg = LinearRegression()
        reg.fit(training_data, training_labels)

        for row, testing_key in enumerate(temporal_decoding.keys()):
            testing_data = np.array(temporal_decoding[testing_key])
            testing_labels = np.array(temporal_labels_partial_num[testing_key])

            score = reg.score(testing_data, testing_labels)

            partial_num_score[row,col] = score
    except:
        for row, testing_key in enumerate(temporal_decoding.keys()):
            partial_num_score[row,col] = 0

plt.title('Confusion Matrix of Partial Number Memory')
plt.xlabel('Trained On')
plt.ylabel('Testing On')
plt.xticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.yticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.imshow(partial_num_score, origin='lower', vmin=0,vmax=1)
plt.colorbar()
plt.show()



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

reg = LinearRegression()
reg.fit(individual_pca_temporal_num_1[3],temporal_labels_first_num[3])

print('Number 1 on Number 1')
for key in individual_pca_temporal_num_1.keys():
    print(reg.score(individual_pca_temporal_num_1[key],temporal_labels_first_num[key]))
print('Number 1 on Number 2')
for key in individual_pca_temporal_num_1.keys():
    print(reg.score(individual_pca_temporal_num_1[key],temporal_labels_second_num[key]))



pca_one_scores = np.zeros((len(temporal_decoding.keys()),len(temporal_decoding.keys())))

for col, training_key in enumerate(temporal_decoding.keys()):
    training_data = np.array(individual_pca_temporal_num_1[training_key])
    training_labels = np.array(temporal_labels_first_num[training_key])
    try:
        reg = LinearRegression()
        reg.fit(training_data, training_labels)

        for row, testing_key in enumerate(temporal_decoding.keys()):
            testing_data = np.array(individual_pca_temporal_num_1[testing_key])
            testing_labels = np.array(temporal_labels_first_num[testing_key])

            score = reg.score(testing_data, testing_labels)

            pca_one_scores[row,col] = score
    except:
        for row, testing_key in enumerate(temporal_decoding.keys()):
            pca_one_scores[row,col] = 0

plt.title('Confusion Matrix of First Num PCA to First Num Label')
plt.xlabel('Trained On')
plt.ylabel('Testing On')
plt.xticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.yticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.imshow(pca_one_scores, origin='lower', vmin=0,vmax=1)
plt.colorbar()
plt.show()



pca_one_scores = np.zeros((len(temporal_decoding.keys()),len(temporal_decoding.keys())))

for col, training_key in enumerate(temporal_decoding.keys()):
    training_data = np.array(individual_pca_temporal_num_1[training_key])
    training_labels = np.array(temporal_labels_second_num[training_key])
    try:
        reg = LinearRegression()
        reg.fit(training_data, training_labels)

        for row, testing_key in enumerate(temporal_decoding.keys()):
            testing_data = np.array(individual_pca_temporal_num_1[testing_key])
            testing_labels = np.array(temporal_labels_second_num[testing_key])

            score = reg.score(testing_data, testing_labels)

            pca_one_scores[row,col] = score
    except:
        for row, testing_key in enumerate(temporal_decoding.keys()):
            pca_one_scores[row,col] = 0

plt.title('Confusion Matrix of First Num PCA to Second Num Label')
plt.xlabel('Trained On')
plt.ylabel('Testing On')
plt.xticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.yticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.imshow(pca_one_scores, origin='lower', vmin=0,vmax=1)
plt.colorbar()
plt.show()


pca_one_scores = np.zeros((len(temporal_decoding.keys()),len(temporal_decoding.keys())))

for col, training_key in enumerate(temporal_decoding.keys()):
    training_data = np.array(individual_pca_temporal_num_1[training_key])
    training_labels = np.array(temporal_labels_running_sum[training_key])
    try:
        reg = LinearRegression()
        reg.fit(training_data, training_labels)

        for row, testing_key in enumerate(temporal_decoding.keys()):
            testing_data = np.array(individual_pca_temporal_num_1[testing_key])
            testing_labels = np.array(temporal_labels_running_sum[testing_key])

            score = reg.score(testing_data, testing_labels)

            pca_one_scores[row,col] = score
    except:
        for row, testing_key in enumerate(temporal_decoding.keys()):
            pca_one_scores[row,col] = 0

plt.title('Confusion Matrix of First Num PCA to Run Sum Label')
plt.xlabel('Trained On')
plt.ylabel('Testing On')
plt.xticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.yticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.imshow(pca_one_scores, origin='lower', vmin=0,vmax=1)
plt.colorbar()
plt.show()



# number_2
individual_pca_temporal_num_2 = {}
temp_array = np.array(temporal_decoding[0])
for key in range(5,9):
    temp_array = np.vstack((temp_array,np.array(temporal_decoding[key])))

pca_num_2.fit(temp_array)
for key in temporal_decoding.keys():
    if len(temporal_decoding[key]) > 0:
        individual_pca_temporal_num_2[key] = pca_num_2.transform(np.array(temporal_decoding[key]))

reg = LinearRegression()
reg.fit(individual_pca_temporal_num_2[8],temporal_labels_second_num[8])

print('Number 2 on Number 2')
for key in individual_pca_temporal_num_2.keys():
    print(reg.score(individual_pca_temporal_num_2[key],temporal_labels_second_num[key]))
print('Number 2 on Number 1')
for key in individual_pca_temporal_num_2.keys():
    print(reg.score(individual_pca_temporal_num_2[key],temporal_labels_first_num[key]))


pca_two_scores = np.zeros((len(temporal_decoding.keys()),len(temporal_decoding.keys())))

for col, training_key in enumerate(temporal_decoding.keys()):
    training_data = np.array(individual_pca_temporal_num_2[training_key])
    training_labels = np.array(temporal_labels_second_num[training_key])
    try:
        reg = LinearRegression()
        reg.fit(training_data, training_labels)

        for row, testing_key in enumerate(temporal_decoding.keys()):
            testing_data = np.array(individual_pca_temporal_num_2[testing_key])
            testing_labels = np.array(temporal_labels_second_num[testing_key])

            score = reg.score(testing_data, testing_labels)

            pca_two_scores[row,col] = score
    except:
        for row, testing_key in enumerate(temporal_decoding.keys()):
            pca_one_scores[row,col] = 0

plt.title('Confusion Matrix of Second Num PCA')
plt.xlabel('Trained On')
plt.ylabel('Testing On')
plt.xticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.yticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.imshow(pca_two_scores, origin='lower', vmin=0,vmax=1)
plt.colorbar()
plt.show()




reg = LinearRegression()
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

reg = LinearRegression()
reg.fit(temp_array, temp_labels)

print('Running Sum Number 1 on All')
for key in individual_pca_temporal_all.keys():
    print(reg.score(individual_pca_temporal_all[key],temporal_labels_running_sum[key]))


temp_array = np.array(individual_pca_temporal_all[5])
temp_labels = list(temporal_labels_running_sum[5])
for key in range(6,9):
    temp_array = np.vstack((temp_array,np.array(individual_pca_temporal_all[key])))
    temp_labels.extend(temporal_labels_running_sum[key])

reg = LinearRegression()
reg.fit(temp_array, temp_labels)

print('Running Sum Number 2 on All')
for key in individual_pca_temporal_all.keys():
    print(reg.score(individual_pca_temporal_all[key],temporal_labels_running_sum[key]))



pca_run_scores = np.zeros((len(temporal_decoding.keys()),len(temporal_decoding.keys())))

for col, training_key in enumerate(temporal_decoding.keys()):
    training_data = np.array(individual_pca_temporal_all[training_key])
    training_labels = np.array(temporal_labels_running_sum[training_key])
    try:
        reg = LinearRegression()
        reg.fit(training_data, training_labels)

        for row, testing_key in enumerate(temporal_decoding.keys()):
            testing_data = np.array(individual_pca_temporal_all[testing_key])
            testing_labels = np.array(temporal_labels_running_sum[testing_key])

            score = reg.score(testing_data, testing_labels)

            pca_run_scores[row,col] = score
    except:
        for row, testing_key in enumerate(temporal_decoding.keys()):
            pca_one_scores[row,col] = 0

plt.title('Confusion Matrix of Running Sum PCA')
plt.xlabel('Trained On')
plt.ylabel('Testing On')
plt.xticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.yticks(range(len(temporal_decoding.keys())),['N1_S','N1_H','N1_T','N1_O','+','N2_S','N2_H','N2_T','N2_O','='])
plt.imshow(pca_run_scores, origin='lower', vmin=0,vmax=1)
plt.colorbar()
plt.show()
