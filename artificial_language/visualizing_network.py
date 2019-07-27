import torch.nn as nn
import torch

import Gated_LSTM

import numpy as np
import random

from tqdm import tqdm

import matplotlib.pyplot as plt

import utils

from sklearn.decomposition import PCA

dataset_file = 'datasets/a2b_testing.txt'
rules_file = 'language_rules/a2b.rul'
num_layers = 1
hidden_size = 3
PATH = 'models/model_l_1_h_3_ep_10'

vocabulary = set()
with open(rules_file) as rule_file:
    for line_idx, line in enumerate(rule_file):
        line = line.strip()
        if line_idx == 0:
            split_line = line.split('::')
            if split_line[0] != 'vocab':
                print('Ruleset ill-defined. Check line 0 has format "vocab::{V}"')
                raise ValueError
            else:
                elements = split_line[1].replace('}','').replace('{','').strip().split(',')

                vocabulary = set(elements)
vocabulary = sorted(vocabulary)
input_size = len(vocabulary)


test_dataset = []
key_string = 'ABCD'
options = list(set(vocabulary).difference('A'))

for i in range(100):
    test_dataset.append(''.join(np.random.choice(options,3))+key_string+ ''.join(np.random.choice(options,3)))



model = Gated_LSTM.GatedLSTM(input_size,hidden_size)
model.load_state_dict(torch.load(PATH))
model.eval()

fgates = [None]*len(test_dataset[0])
igates = [None]*len(test_dataset[0])
ogates = [None]*len(test_dataset[0])
ggates = [None]*len(test_dataset[0])
hgates = [None]*len(test_dataset[0])
cgates = [None]*len(test_dataset[0])

for line_index, line in enumerate(tqdm(test_dataset)):
    line_length = len(line)
    x = []
    labels = []

    for i in range(line_length):
        current, character_idx = utils.character_to_vector(line[i], vocabulary)
        x.append(current)
        labels.append(character_idx)

    inputs = labels[:line_length-1]
    labels = torch.Tensor(labels[1:]).long()
    x = torch.Tensor(x).reshape(1,line_length,-1)
    x.requires_grad_(True)
    output, hidden, (forget_gates, input_gates, output_gates, intermediate_states, cell_states) = model(x,None)

    decoded = model.decoder(output[:,:line_length-1,:])
    predictions = nn.functional.log_softmax(decoded[0]).reshape(line_length-1,-1)
    probabilities = torch.exp(predictions)

    total_prob = torch.sum(probabilities,1)

    classes = predictions.argmax(1)


    for gate_index in range(len(forget_gates)):
        if fgates[gate_index] is None:
            fgates[gate_index] = []
        if igates[gate_index] is None:
            igates[gate_index] = []
        if ogates[gate_index] is None:
            ogates[gate_index] = []
        if ggates[gate_index] is None:
            ggates[gate_index] = []
        if hgates[gate_index] is None:
            hgates[gate_index] = []
        if cgates[gate_index] is None:
            cgates[gate_index] = []


        fgates[gate_index].append(forget_gates[gate_index].detach().numpy().reshape(1,-1))
        igates[gate_index].append(input_gates[gate_index].detach().numpy().reshape(1,-1))
        ogates[gate_index].append(output_gates[gate_index].detach().numpy().reshape(1,-1))
        ggates[gate_index].append(intermediate_states[gate_index].detach().numpy().reshape(1,-1))
        cgates[gate_index].append(cell_states[gate_index].detach().numpy().reshape(1,-1))
        hgates[gate_index].append(output[:,gate_index,:].detach().numpy().reshape(1,-1))



fig, axs = plt.subplots(6)

color_dict = {0:'#1abc9c',1:'#2ecc71',2:'#3498db',3:'#9b59b6',4:'#34495e',5:'#f1c40f',6:'#e67e22',7:'#e74c3c'}
locations = []

# Forget Gate
all_fgates = []
average_activity = []
for gate_index in range(len(fgates)):
    avg_gate = np.array(fgates[gate_index]).reshape(len(test_dataset),-1).mean(0)

    all_fgates.extend(fgates[gate_index])
    locations.extend([gate_index]*len(fgates[gate_index]))
    average_activity.append(avg_gate)

average_activity = np.array(average_activity)
all_fgates = np.array(all_fgates).reshape(len(fgates)*len(test_dataset),-1)

mean_gate = all_fgates.mean(0)
std_gate = all_fgates.std(0)

zscore_activity = []
for gate_index in range(len(fgates)):
    avg_gate = (np.array(fgates[gate_index]).reshape(len(test_dataset),-1) - mean_gate)/(std_gate)
    zscore_activity.append(avg_gate.mean(0))

zscore_activity = np.array(zscore_activity)

axs[0].imshow(zscore_activity.T)
axs[0].set_ylabel('F')
axs[0].set_xticks([])

# Input Gate
all_igates = []
average_activity = []
for gate_index in range(len(igates)):
    avg_gate = np.array(igates[gate_index]).reshape(len(test_dataset),-1).mean(0)

    all_igates.extend(igates[gate_index])
    average_activity.append(avg_gate)

average_activity = np.array(average_activity)
all_igates = np.array(all_igates).reshape(len(igates)*len(test_dataset),-1)

mean_gate = all_igates.mean(0)
std_gate = all_igates.std(0)

zscore_activity = []
for gate_index in range(len(igates)):
    avg_gate = (np.array(igates[gate_index]).reshape(len(test_dataset),-1) - mean_gate)/(std_gate)
    zscore_activity.append(avg_gate.mean(0))

zscore_activity = np.array(zscore_activity)

axs[1].imshow(zscore_activity.T)
axs[1].set_ylabel('I')
axs[1].set_xticks([])

# Output Gate
all_ogates = []
average_activity = []
for gate_index in range(len(ogates)):
    avg_gate = np.array(ogates[gate_index]).reshape(len(test_dataset),-1).mean(0)

    all_ogates.extend(ogates[gate_index])
    average_activity.append(avg_gate)

average_activity = np.array(average_activity)
all_ogates = np.array(all_ogates).reshape(len(ogates)*len(test_dataset),-1)

mean_gate = all_ogates.mean(0)
std_gate = all_ogates.std(0)

zscore_activity = []
for gate_index in range(len(ogates)):
    avg_gate = (np.array(ogates[gate_index]).reshape(len(test_dataset),-1) - mean_gate)/(std_gate)
    zscore_activity.append(avg_gate.mean(0))

zscore_activity = np.array(zscore_activity)

axs[2].imshow(zscore_activity.T)
axs[2].set_ylabel('O')
axs[2].set_xticks([])

# Intermediate Gate
all_ggates = []
average_activity = []
for gate_index in range(len(ggates)):
    avg_gate = np.array(ggates[gate_index]).reshape(len(test_dataset),-1).mean(0)

    all_ggates.extend(ggates[gate_index])
    average_activity.append(avg_gate)

average_activity = np.array(average_activity)
all_ggates = np.array(all_ggates).reshape(len(ggates)*len(test_dataset),-1)

mean_gate = all_ggates.mean(0)
std_gate = all_ggates.std(0)

zscore_activity = []
for gate_index in range(len(ggates)):
    avg_gate = (np.array(ggates[gate_index]).reshape(len(test_dataset),-1) - mean_gate)/(std_gate)
    zscore_activity.append(avg_gate.mean(0))

zscore_activity = np.array(zscore_activity)

axs[3].imshow(zscore_activity.T)
axs[3].set_ylabel('G')
axs[3].set_xticks([])

# Cell State
all_cgates = []
average_activity = []
for gate_index in range(len(cgates)):
    avg_gate = np.array(cgates[gate_index]).reshape(len(test_dataset),-1).mean(0)

    all_cgates.extend(cgates[gate_index])
    average_activity.append(avg_gate)

average_activity = np.array(average_activity)
all_cgates = np.array(all_cgates).reshape(len(cgates)*len(test_dataset),-1)

mean_gate = all_cgates.mean(0)
std_gate = all_cgates.std(0)

zscore_activity = []
for gate_index in range(len(cgates)):
    avg_gate = (np.array(cgates[gate_index]).reshape(len(test_dataset),-1) - mean_gate)/(std_gate)
    zscore_activity.append(avg_gate.mean(0))

zscore_activity = np.array(zscore_activity)

axs[4].imshow(zscore_activity.T)
axs[4].set_ylabel('C')
axs[4].set_xticks([])


# Hidden State
all_hgates = []
average_activity = []
for gate_index in range(len(hgates)):
    avg_gate = np.array(hgates[gate_index]).reshape(len(test_dataset),-1).mean(0)

    all_hgates.extend(hgates[gate_index])
    average_activity.append(avg_gate)

average_activity = np.array(average_activity)
all_hgates = np.array(all_hgates).reshape(len(hgates)*len(test_dataset),-1)

mean_gate = all_hgates.mean(0)
std_gate = all_hgates.std(0)

zscore_activity = []
for gate_index in range(len(hgates)):
    avg_gate = (np.array(hgates[gate_index]).reshape(len(test_dataset),-1) - mean_gate)/(std_gate)
    zscore_activity.append(avg_gate.mean(0))

zscore_activity = np.array(zscore_activity)

axs[5].imshow(zscore_activity.T)
axs[5].set_ylabel('H')
axs[5].set_xticks([])

plt.show()
plt.close()

pca = PCA(n_components=2)
pcaf = pca.fit_transform(all_fgates)

pca = PCA(n_components=2)
pcai = pca.fit_transform(all_igates)

pca = PCA(n_components=2)
pcao = pca.fit_transform(all_ogates)

pca = PCA(n_components=2)
pcag = pca.fit_transform(all_ggates)

pca = PCA(n_components=2)
pcac = pca.fit_transform(all_cgates)

pca = PCA(n_components=2)
pcah = pca.fit_transform(all_hgates)

seen = set()
fig, axs = plt.subplots(6)
for idx in range(len(all_fgates)):
    if locations[idx] not in seen:
        axs[0].scatter(pcaf.T[0][idx],pcaf.T[1][idx], c = color_dict[locations[idx]], label=locations[idx])
        axs[1].scatter(pcai.T[0][idx],pcai.T[1][idx], c = color_dict[locations[idx]])
        axs[2].scatter(pcao.T[0][idx],pcao.T[1][idx], c = color_dict[locations[idx]])
        axs[3].scatter(pcag.T[0][idx],pcag.T[1][idx], c = color_dict[locations[idx]])
        axs[4].scatter(pcac.T[0][idx],pcac.T[1][idx], c = color_dict[locations[idx]])
        axs[5].scatter(pcah.T[0][idx],pcah.T[1][idx], c = color_dict[locations[idx]])
        seen.add(locations[idx])
    else:
        axs[0].scatter(pcaf.T[0][idx],pcaf.T[1][idx], c = color_dict[locations[idx]])
        axs[1].scatter(pcai.T[0][idx],pcai.T[1][idx], c = color_dict[locations[idx]])
        axs[2].scatter(pcao.T[0][idx],pcao.T[1][idx], c = color_dict[locations[idx]])
        axs[3].scatter(pcag.T[0][idx],pcag.T[1][idx], c = color_dict[locations[idx]])
        axs[4].scatter(pcac.T[0][idx],pcac.T[1][idx], c = color_dict[locations[idx]])
        axs[5].scatter(pcah.T[0][idx],pcah.T[1][idx], c = color_dict[locations[idx]])

axs[0].set_xticks([])
axs[1].set_xticks([])
axs[2].set_xticks([])
axs[3].set_xticks([])
axs[4].set_xticks([])
axs[5].set_xticks([])

axs[0].set_yticks([])
axs[1].set_yticks([])
axs[2].set_yticks([])
axs[3].set_yticks([])
axs[4].set_yticks([])
axs[5].set_yticks([])


axs[0].set_ylabel('F')
axs[1].set_ylabel('I')
axs[2].set_ylabel('O')
axs[3].set_ylabel('G')
axs[4].set_ylabel('C')
axs[5].set_ylabel('H')

fig.legend(numpoints=1)

plt.show()
plt.close()
