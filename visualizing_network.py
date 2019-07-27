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


dataset_training = 'datasets/arithmetic/1e2/training.txt'
dataset_testing = 'datasets/arithmetic/1e2/testing.txt'

dataset = Dataset(dataset_training,dataset_testing)

num_layers = 1
hidden_size = 100
num_epochs = 5
input_size = dataset.vector_size
PATH = 'arithmetic_l_'+str(num_layers)+'_h_'+str(hidden_size)+'_ep_'+str(num_epochs)

model = GatedGRU(dataset.vector_size,hidden_size,output_size=1)
model.load_state_dict(torch.load(PATH))
model.eval()

ugates_addition = [None]*5
rgates_addition = [None]*5
hgates_addition = [None]*5

ugates_equals = [None]*3
rgates_equals = [None]*3
hgates_equals = [None]*3

ugates_categorical = [None]*4
rgates_categorical = [None]*4
hgates_categorical = [None]*4

all_ugates = []
all_rgates = []
all_hidden = []

num_1_count = 0
num_2_count = 0

location = []

for idx in range(dataset.testing_size()):
    input, label, line = dataset.testing_item(idx)

    addition_index = line.index('+')

    equals_index = line.index('=')

    x = torch.Tensor(input).reshape(1, -1, dataset.vector_size)

    hidden = model.init_hidden()

    decoded, hidden, (update_gates, reset_gates, hidden_states) = model(x,hidden)

    for i in range(len(update_gates)):
        all_ugates.append(update_gates[i][0][0].detach().numpy())
        all_rgates.append(reset_gates[i][0][0].detach().numpy())
        all_hidden.append(hidden_states[i][0][0].detach().numpy())

    # Used for addition analysis
    for mod_idx in range(-2,3,1):
        rel_idx = mod_idx + addition_index

        if hgates_addition[mod_idx+2] is None:
            hgates_addition[mod_idx+2] = hidden_states[rel_idx][0][0].detach().numpy()
        else:
            hgates_addition[mod_idx+2] = (hgates_addition[mod_idx+2]*idx + hidden_states[rel_idx][0][0].detach().numpy())/(idx+1)

        if ugates_addition[mod_idx+2] is None:
            ugates_addition[mod_idx+2] = update_gates[rel_idx][0][0].detach().numpy()
        else:
            ugates_addition[mod_idx+2] = (ugates_addition[mod_idx+2]*idx + update_gates[rel_idx][0][0].detach().numpy())/(idx+1)

        if rgates_addition[mod_idx+2] is None:
            rgates_addition[mod_idx+2] = reset_gates[rel_idx][0][0].detach().numpy()
        else:
            rgates_addition[mod_idx+2] = (rgates_addition[mod_idx+2]*idx + reset_gates[rel_idx][0][0].detach().numpy())/(idx+1)

    # used for equals analysis
    for mod_idx in range(-2,1,1):
        rel_idx = mod_idx + equals_index

        if hgates_equals[mod_idx+2] is None:
            hgates_equals[mod_idx+2] = hidden_states[rel_idx][0][0].detach().numpy()
        else:
            hgates_equals[mod_idx+2] = (hgates_equals[mod_idx+2]*idx + hidden_states[rel_idx][0][0].detach().numpy())/(idx+1)

        if ugates_equals[mod_idx+2] is None:
            ugates_equals[mod_idx+2] = update_gates[rel_idx][0][0].detach().numpy()
        else:
            ugates_equals[mod_idx+2] = (ugates_equals[mod_idx+2]*idx + update_gates[rel_idx][0][0].detach().numpy())/(idx+1)

        if rgates_equals[mod_idx+2] is None:
            rgates_equals[mod_idx+2] = reset_gates[rel_idx][0][0].detach().numpy()
        else:
            rgates_equals[mod_idx+2] = (rgates_equals[mod_idx+2]*idx + reset_gates[rel_idx][0][0].detach().numpy())/(idx+1)

    # used for categorical analysis
    for i in range(len(update_gates)):

        if i < addition_index:
            modification_index = 0
            if hgates_categorical[modification_index] is None:
                hgates_categorical[modification_index] = hidden_states[i][0][0].detach().numpy()
            else:
                hgates_categorical[modification_index] = (hgates_categorical[modification_index]*num_1_count + hidden_states[i][0][0].detach().numpy())/(num_1_count+1)

            if ugates_categorical[modification_index] is None:
                ugates_categorical[modification_index] = update_gates[i][0][0].detach().numpy()
            else:
                ugates_categorical[modification_index] = (ugates_categorical[modification_index]*num_1_count + update_gates[i][0][0].detach().numpy())/(num_1_count+1)

            if rgates_categorical[modification_index] is None:
                rgates_categorical[modification_index] = reset_gates[i][0][0].detach().numpy()
            else:
                rgates_categorical[modification_index] = (rgates_categorical[modification_index]*num_1_count + reset_gates[i][0][0].detach().numpy())/(num_1_count+1)

            num_1_count += 1

            location.append(modification_index)

        elif i == addition_index:
            modification_index = 1
            if hgates_categorical[modification_index] is None:
                hgates_categorical[modification_index] = hidden_states[i][0][0].detach().numpy()
            else:
                hgates_categorical[modification_index] = (hgates_categorical[modification_index]*idx + hidden_states[i][0][0].detach().numpy())/(idx+1)

            if ugates_categorical[modification_index] is None:
                ugates_categorical[modification_index] = update_gates[i][0][0].detach().numpy()
            else:
                ugates_categorical[modification_index] = (ugates_categorical[modification_index]*idx + update_gates[i][0][0].detach().numpy())/(idx+1)

            if rgates_categorical[modification_index] is None:
                rgates_categorical[modification_index] = reset_gates[i][0][0].detach().numpy()
            else:
                rgates_categorical[modification_index] = (rgates_categorical[modification_index]*idx + reset_gates[i][0][0].detach().numpy())/(idx+1)

            location.append(modification_index)

        elif addition_index < i < equals_index:
            modification_index = 2
            if hgates_categorical[modification_index] is None:
                hgates_categorical[modification_index] = hidden_states[i][0][0].detach().numpy()
            else:
                hgates_categorical[modification_index] = (hgates_categorical[modification_index]*num_2_count + hidden_states[i][0][0].detach().numpy())/(num_2_count+1)

            if ugates_categorical[modification_index] is None:
                ugates_categorical[modification_index] = update_gates[i][0][0].detach().numpy()
            else:
                ugates_categorical[modification_index] = (ugates_categorical[modification_index]*num_2_count + update_gates[i][0][0].detach().numpy())/(num_2_count+1)

            if rgates_categorical[modification_index] is None:
                rgates_categorical[modification_index] = reset_gates[i][0][0].detach().numpy()
            else:
                rgates_categorical[modification_index] = (rgates_categorical[modification_index]*num_2_count + reset_gates[i][0][0].detach().numpy())/(num_2_count+1)

            num_2_count += 1

            location.append(modification_index)
        elif i == equals_index:
            modification_index = 3
            if hgates_categorical[modification_index] is None:
                hgates_categorical[modification_index] = hidden_states[i][0][0].detach().numpy()
            else:
                hgates_categorical[modification_index] = (hgates_categorical[modification_index]*idx + hidden_states[i][0][0].detach().numpy())/(idx+1)

            if ugates_categorical[modification_index] is None:
                ugates_categorical[modification_index] = update_gates[i][0][0].detach().numpy()
            else:
                ugates_categorical[modification_index] = (ugates_categorical[modification_index]*idx + update_gates[i][0][0].detach().numpy())/(idx+1)

            if rgates_categorical[modification_index] is None:
                rgates_categorical[modification_index] = reset_gates[i][0][0].detach().numpy()
            else:
                rgates_categorical[modification_index] = (rgates_categorical[modification_index]*idx + reset_gates[i][0][0].detach().numpy())/(idx+1)

            location.append(modification_index)




all_ugates = np.array(all_ugates)
all_rgates = np.array(all_rgates)
all_hidden = np.array(all_hidden)


pcah = PCA(n_components=min(hidden_size,5))
pcahgates = pcah.fit_transform(all_hidden)
pcah_var = np.sum(pcah.explained_variance_ratio_)

pcau = PCA(n_components=min(hidden_size,5))
pcaugates = pcau.fit_transform(all_ugates)
pcau_var = np.sum(pcau.explained_variance_ratio_)

pcar = PCA(n_components=min(hidden_size,5))
pcargates = pcar.fit_transform(all_rgates)
pcar_var = np.sum(pcar.explained_variance_ratio_)

yticks = []
for i in range(min(hidden_size,5)):
    yticks.append('PC'+str(i+1))


color_dict = {0:'#1abc9c',1:'#2ecc71',2:'#3498db',3:'#9b59b6',4:'#34495e',5:'#f1c40f',6:'#e67e22',7:'#e74c3c'}
locations = []

# Near + sign

# update gate
mean_gate = pcaugates.mean(0)
std_gate = pcaugates.std(0)
zscore_activity = []
for gate_index in range(len(ugates_addition)):
    transformed = pcau.transform(ugates_addition[gate_index].reshape(1,-1))
    zscore_activity.append(((transformed - mean_gate)/std_gate)[0])

uzscore_activity = np.array(zscore_activity)


mean_gate = pcahgates.mean(0)
std_gate = pcahgates.std(0)
zscore_activity = []
for gate_index in range(len(hgates_addition)):
    transformed = pcah.transform(hgates_addition[gate_index].reshape(1,-1))
    zscore_activity.append(((transformed - mean_gate)/std_gate)[0])

hzscore_activity = np.array(zscore_activity)


mean_gate = pcargates.mean(0)
std_gate = pcargates.std(0)
zscore_activity = []
for gate_index in range(len(rgates_addition)):
    transformed = pcar.transform(rgates_addition[gate_index].reshape(1,-1))
    zscore_activity.append(((transformed - mean_gate)/std_gate)[0])

rzscore_activity = np.array(zscore_activity)

# PCA analysis
fig, axs = plt.subplots(3)
fig.suptitle('L:'+str(num_layers)+' H:'+str(hidden_size)+'\nPCAd Gates\n Near + ')
ugates = pcau.transform(ugates_addition)
hgates = pcah.transform(hgates_addition)
rgates = pcar.transform(rgates_addition)

vmax = max(np.max(np.abs(ugates)),np.max(np.abs(hgates)),np.max(np.abs(rgates)))

axs[0].imshow(ugates.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[0].set_ylabel('Update\nVar: '+str(round(pcau_var,2)))
axs[0].set_xticks([])
axs[0].set_yticks(range(len(yticks)))
axs[0].set_yticklabels(yticks)

axs[1].imshow(rgates.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[1].set_ylabel('Reset\nVar: '+str(round(pcar_var,2)))
axs[1].set_xticks([])
axs[1].set_yticks(range(len(yticks)))
axs[1].set_yticklabels(yticks)


im2 = axs[2].imshow(hgates.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[2].set_ylabel('Hidden\nVar: '+str(round(pcah_var,2)))
axs[2].set_xticks(range(len(hgates)))
axs[2].set_xticklabels(['','End','+','New',''])
axs[2].set_yticks(range(len(yticks)))
axs[2].set_yticklabels(yticks)

cbar = fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.95)



fig, axs = plt.subplots(3)
vmax = max(np.max(np.abs(uzscore_activity)),np.max(np.abs(rzscore_activity)),np.max(np.abs(hzscore_activity)))

axs[0].imshow(uzscore_activity.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[0].set_ylabel('Update\nVar: '+str(round(pcau_var,2)))
axs[0].set_xticks([])
axs[0].set_yticks(range(len(yticks)))
axs[0].set_yticklabels(yticks)

axs[1].imshow(rzscore_activity.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[1].set_ylabel('Reset\nVar: '+str(round(pcar_var,2)))
axs[1].set_xticks([])
axs[1].set_yticks(range(len(yticks)))
axs[1].set_yticklabels(yticks)

im2 = axs[2].imshow(hzscore_activity.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[2].set_ylabel('Hidden\nVar: '+str(round(pcah_var,2)))
axs[2].set_xticks(range(len(hgates)))
axs[2].set_xticklabels(['','End','+','New',''])
axs[2].set_yticks(range(len(yticks)))
axs[2].set_yticklabels(yticks)

cbar = fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.95)
fig.suptitle('L:'+str(num_layers)+' H:'+str(hidden_size)+'\nPCAd Z-Score\n Gates Near + ')



fig, axs = plt.subplots(3)

axs[0].imshow(np.abs(uzscore_activity).T,norm= Normalize(vmin=0, vmax=vmax), cmap='Reds')
axs[0].set_ylabel('Update\nVar: '+str(round(pcau_var,2)))
axs[0].set_xticks([])
axs[0].set_yticks(range(len(yticks)))
axs[0].set_yticklabels(yticks)

axs[1].imshow(np.abs(rzscore_activity).T,norm= Normalize(vmin=0, vmax=vmax), cmap='Reds')
axs[1].set_ylabel('Reset\nVar: '+str(round(pcar_var,2)))
axs[1].set_xticks([])
axs[1].set_yticks(range(len(yticks)))
axs[1].set_yticklabels(yticks)

im2 = axs[2].imshow(np.abs(hzscore_activity).T,norm= Normalize(vmin=0, vmax=vmax), cmap='Reds')
axs[2].set_ylabel('Hidden\nVar: '+str(round(pcah_var,2)))
axs[2].set_xticks(range(len(hgates)))
axs[2].set_xticklabels(['','End','+','New',''])
axs[2].set_yticks(range(len(yticks)))
axs[2].set_yticklabels(yticks)

cbar = fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.95)
fig.suptitle('L:'+str(num_layers)+' H:'+str(hidden_size)+'\nPCAd Absolute Z-Score\n Gates Near + ')




# Equal sign analysis

# update gate
mean_gate = pcaugates.mean(0)
std_gate = pcaugates.std(0)
zscore_activity = []
for gate_index in range(len(ugates_equals)):
    transformed = pcau.transform(ugates_equals[gate_index].reshape(1,-1))
    zscore_activity.append(((transformed - mean_gate)/std_gate)[0])

uzscore_activity = np.array(zscore_activity)


mean_gate = pcahgates.mean(0)
std_gate = pcahgates.std(0)
zscore_activity = []
for gate_index in range(len(hgates_equals)):
    transformed = pcah.transform(hgates_equals[gate_index].reshape(1,-1))
    zscore_activity.append(((transformed - mean_gate)/std_gate)[0])

hzscore_activity = np.array(zscore_activity)


mean_gate = pcargates.mean(0)
std_gate = pcargates.std(0)
zscore_activity = []
for gate_index in range(len(rgates_equals)):
    transformed = pcar.transform(rgates_equals[gate_index].reshape(1,-1))
    zscore_activity.append(((transformed - mean_gate)/std_gate)[0])

rzscore_activity = np.array(zscore_activity)

color_dict = {0:'#1abc9c',1:'#2ecc71',2:'#3498db',3:'#9b59b6',4:'#34495e',5:'#f1c40f',6:'#e67e22',7:'#e74c3c'}
locations = []


fig, axs = plt.subplots(3)
fig.suptitle('L:'+str(num_layers)+' H:'+str(hidden_size)+'\nPCAd Gates\n Near = ')
ugates = pcau.transform(ugates_equals)
hgates = pcah.transform(hgates_equals)
rgates = pcar.transform(rgates_equals)

vmax = max(np.max(np.abs(ugates)),np.max(np.abs(hgates)),np.max(np.abs(rgates)))

axs[0].imshow(ugates.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[0].set_ylabel('Update\nVar: '+str(round(pcau_var,2)))
axs[0].set_xticks([])
axs[0].set_yticks(range(len(yticks)))
axs[0].set_yticklabels(yticks)

axs[1].imshow(rgates.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[1].set_ylabel('Reset\nVar: '+str(round(pcar_var,2)))
axs[1].set_xticks([])
axs[1].set_yticks(range(len(yticks)))
axs[1].set_yticklabels(yticks)

im2 = axs[2].imshow(hgates.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[2].set_ylabel('Hidden\nVar: '+str(round(pcah_var,2)))
axs[2].set_xticks(range(len(hgates)))
axs[2].set_xticklabels(['','End','='])
axs[2].set_yticks(range(len(yticks)))
axs[2].set_yticklabels(yticks)

cbar = fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.95)



fig, axs = plt.subplots(3)
fig.suptitle('L:'+str(num_layers)+' H:'+str(hidden_size)+'\nPCAd Z-Score\n Gates Near = ')

vmax = max(np.max(np.abs(uzscore_activity)),np.max(np.abs(rzscore_activity)),np.max(np.abs(hzscore_activity)))

axs[0].imshow(uzscore_activity.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[0].set_ylabel('Update\nVar: '+str(round(pcau_var,2)))
axs[0].set_xticks([])
axs[0].set_yticks(range(len(yticks)))
axs[0].set_yticklabels(yticks)

axs[1].imshow(rzscore_activity.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[1].set_ylabel('Reset\nVar: '+str(round(pcar_var,2)))
axs[1].set_xticks([])
axs[1].set_yticks(range(len(yticks)))
axs[1].set_yticklabels(yticks)

im2 = axs[2].imshow(hzscore_activity.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[2].set_ylabel('Hidden\nVar: '+str(round(pcah_var,2)))
axs[2].set_xticks(range(len(hgates)))
axs[2].set_xticklabels(['','End','='])
axs[2].set_yticks(range(len(yticks)))
axs[2].set_yticklabels(yticks)

cbar = fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.95)


fig, axs = plt.subplots(3)
fig.suptitle('L:'+str(num_layers)+' H:'+str(hidden_size)+'\nPCAd Absolute Z-Score\n Gates Near = ')
axs[0].imshow(np.abs(uzscore_activity).T,norm= Normalize(vmin=0, vmax=vmax), cmap='Reds')
axs[0].set_ylabel('Update\nVar: '+str(round(pcau_var,2)))
axs[0].set_xticks([])
axs[0].set_yticks(range(len(yticks)))
axs[0].set_yticklabels(yticks)

axs[1].imshow(np.abs(rzscore_activity).T,norm= Normalize(vmin=0, vmax=vmax), cmap='Reds')
axs[1].set_ylabel('Reset\nVar: '+str(round(pcar_var,2)))
axs[1].set_xticks([])
axs[1].set_yticks(range(len(yticks)))
axs[1].set_yticklabels(yticks)

im2 = axs[2].imshow(np.abs(hzscore_activity).T,norm= Normalize(vmin=0, vmax=vmax), cmap='Reds')
axs[2].set_ylabel('Hidden\nVar: '+str(round(pcah_var,2)))
axs[2].set_xticks(range(len(hgates)))
axs[2].set_xticklabels(['','End','='])
axs[2].set_yticks(range(len(yticks)))
axs[2].set_yticklabels(yticks)

cbar = fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.95)



# average over all datasets

#n1,+,n2,=

# update gate
mean_gate = pcaugates.mean(0)
std_gate = pcaugates.std(0)
zscore_activity = []
for gate_index in range(len(ugates_categorical)):
    transformed = pcau.transform(ugates_categorical[gate_index].reshape(1,-1))
    zscore_activity.append(((transformed - mean_gate)/std_gate)[0])

uzscore_activity = np.array(zscore_activity)


mean_gate = pcahgates.mean(0)
std_gate = pcahgates.std(0)
zscore_activity = []
for gate_index in range(len(hgates_categorical)):
    transformed = pcah.transform(hgates_categorical[gate_index].reshape(1,-1))
    zscore_activity.append(((transformed - mean_gate)/std_gate)[0])

hzscore_activity = np.array(zscore_activity)


mean_gate = pcargates.mean(0)
std_gate = pcargates.std(0)
zscore_activity = []
for gate_index in range(len(rgates_categorical)):
    transformed = pcar.transform(rgates_categorical[gate_index].reshape(1,-1))
    zscore_activity.append(((transformed - mean_gate)/std_gate)[0])

rzscore_activity = np.array(zscore_activity)


fig, axs = plt.subplots(3)
fig.suptitle('L:'+str(num_layers)+' H:'+str(hidden_size)+'\nPCAd Gates\n Categorically Averaged')
ugates = pcau.transform(ugates_categorical)
hgates = pcah.transform(hgates_categorical)
rgates = pcar.transform(rgates_categorical)

vmax = max(np.max(np.abs(ugates)),np.max(np.abs(hgates)),np.max(np.abs(rgates)))

axs[0].imshow(ugates.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[0].set_ylabel('Update\nVar: '+str(round(pcau_var,2)))
axs[0].set_xticks([])
axs[0].set_yticks(range(len(yticks)))
axs[0].set_yticklabels(yticks)

axs[1].imshow(rgates.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[1].set_ylabel('Reset\nVar: '+str(round(pcar_var,2)))
axs[1].set_xticks([])
axs[1].set_yticks(range(len(yticks)))
axs[1].set_yticklabels(yticks)

im2 = axs[2].imshow(hgates.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[2].set_ylabel('Hidden\nVar: '+str(round(pcah_var,2)))
axs[2].set_xticks(range(len(hgates)))
axs[2].set_xticklabels(['Num 1','+','Num 2','='])
axs[2].set_yticks(range(len(yticks)))
axs[2].set_yticklabels(yticks)

cbar = fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.95)


fig, axs = plt.subplots(3)
fig.suptitle('L:'+str(num_layers)+' H:'+str(hidden_size)+'\nPCAd Z-Score Gates\n Categorically Averaged')

vmax = max(np.max(np.abs(uzscore_activity)),np.max(np.abs(rzscore_activity)),np.max(np.abs(hzscore_activity)))

axs[0].imshow(uzscore_activity.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[0].set_ylabel('Update\nVar: '+str(round(pcau_var,2)))
axs[0].set_xticks([])
axs[0].set_yticks(range(len(yticks)))
axs[0].set_yticklabels(yticks)

axs[1].imshow(rzscore_activity.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[1].set_ylabel('Reset\nVar: '+str(round(pcar_var,2)))
axs[1].set_xticks([])
axs[1].set_yticks(range(len(yticks)))
axs[1].set_yticklabels(yticks)

im2 = axs[2].imshow(hzscore_activity.T,norm= Normalize(vmin=-vmax, vmax=vmax), cmap='seismic')
axs[2].set_ylabel('Hidden\nVar: '+str(round(pcah_var,2)))
axs[2].set_xticks(range(len(hgates)))
axs[2].set_xticklabels(['Num 1','+','Num 2','='])
axs[2].set_yticks(range(len(yticks)))
axs[2].set_yticklabels(yticks)

cbar = fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.95)



fig, axs = plt.subplots(3)
fig.suptitle('L:'+str(num_layers)+' H:'+str(hidden_size)+'\nPCAd Absolute Z-Score Gates\n Categorically Averaged')

axs[0].imshow(np.abs(uzscore_activity).T,norm= Normalize(vmin=0, vmax=vmax), cmap='Reds')
axs[0].set_ylabel('Update\nVar: '+str(round(pcau_var,2)))
axs[0].set_xticks([])
axs[0].set_yticks(range(len(yticks)))
axs[0].set_yticklabels(yticks)

axs[1].imshow(np.abs(rzscore_activity).T,norm= Normalize(vmin=0, vmax=vmax), cmap='Reds')
axs[1].set_ylabel('Reset\nVar: '+str(round(pcar_var,2)))
axs[1].set_xticks([])
axs[1].set_yticks(range(len(yticks)))
axs[1].set_yticklabels(yticks)

im2 = axs[2].imshow(np.abs(hzscore_activity).T,norm= Normalize(vmin=0, vmax=vmax), cmap='Reds')
axs[2].set_ylabel('Hidden\nVar: '+str(round(pcah_var,2)))
axs[2].set_xticks(range(len(hgates)))
axs[2].set_xticklabels(['Num 1','+','Num 2','='])
axs[2].set_yticks(range(len(yticks)))
axs[2].set_yticklabels(yticks)

cbar = fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.95)

plt.show()
plt.close()
