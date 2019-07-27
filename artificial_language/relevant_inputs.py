import torch.nn as nn
import torch

import Gated_LSTM

import numpy as np
import random

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

import utils

from sklearn.decomposition import PCA

np.random.seed(0)

dataset_file = 'datasets/a2bcd_testing.txt'
rules_file = 'language_rules/a2bcd.rul'
num_layers = 1
hidden_size = 3
ABSOLUTE_EFFECT = True
MAGNITUDE = True
PATH = 'models/model_a2bcd_l_1_h_3_ep_50'

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

def perturb(vector,index,quantity):
    new_vector = vector.clone()
    new_vector[0,index] = vector[0,index] + quantity
    return new_vector

test_dataset = []
key_string = 'ABCD'
options = sorted(list(set(vocabulary).difference('A')))

model = Gated_LSTM.GatedLSTM(input_size,hidden_size)
model.load_state_dict(torch.load(PATH))
model.eval()

for i in range(100):
    test_dataset.append(''.join(np.random.choice(options,3))+key_string+ ''.join(np.random.choice(options,3)))

for line_index, line in enumerate(tqdm(test_dataset)):
    print(line)
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

    all_estimates = []
    for word in range(line_length-1):
        estim_df = [0]*(line_length-1)

        sub_x = x[:,:word+1,:]

        sub_x.requires_grad = True


        relevant_indices = inputs[:word+1]
        relevant_labels = labels[:word+1]

        sub_output, sub_hidden, _ = model(sub_x,None)

        sub_decoded = model.decoder(sub_hidden[0])
        sub_predictions = nn.functional.log_softmax(sub_decoded[0]).reshape(len(sub_x),-1)
        sub_probabilities = torch.exp(sub_predictions)

        if word != line_length-2:
            gradient = torch.autograd.grad(outputs=sub_probabilities[0,relevant_labels[len(relevant_labels)-1]],inputs=sub_x,retain_graph=True)[0]
        else:
            gradient = torch.autograd.grad(outputs=sub_probabilities[0,relevant_labels[len(relevant_labels)-1]],inputs=sub_x,retain_graph=False)[0]

        for pert_idx in range(word+1):
            if ABSOLUTE_EFFECT:
                df  = torch.norm(gradient[0, pert_idx, :]).item()
            else:
                df = gradient[0, pert_idx, relevant_indices[pert_idx]]

            estim_df[pert_idx] = df

        # sub_x.grad.zero_()
        # for pert_idx in range(word+1):
        #
        #     perturbation = 0.01
        #
        #     # perturbing
        #     pert_sub_x = sub_x.clone()
        #     new_vector = perturb(sub_x[:,pert_idx,:],relevant_indices[pert_idx],perturbation)
        #     pert_sub_x[:,pert_idx,:] = new_vector
        #
        #     # calculating perturbed output
        #     pert_output, pert_hidden, _ = model(pert_sub_x,None)
        #     pert_decoded = model.decoder(pert_hidden[0])
        #     pert_predictions = nn.functional.log_softmax(pert_decoded[0]).reshape(len(sub_x),-1)
        #     pert_probabilities = torch.exp(pert_predictions)
        #     df = (pert_probabilities[0,relevant_labels[pert_idx].item()].item() - sub_probabilities[0,relevant_labels[pert_idx].item()].item())/perturbation
        #
        #     estim_df[pert_idx] = df
        #

        estim_df = np.array(estim_df)
        if ABSOLUTE_EFFECT:
            estim_df = np.abs(estim_df)
        min_df = np.min(estim_df)
        estim_df = estim_df - min_df+0.000001

        max_df = np.max(estim_df)
        if max_df == 0:
            max_df = 1

        estim_df /= max_df

        all_estimates.append(estim_df)
    all_estimates = np.array(all_estimates)

    if ABSOLUTE_EFFECT:
        cmap = matplotlib.cm.get_cmap('Reds')
    else:
        cmap = matplotlib.cm.get_cmap('Reds')

    for row in range(len(all_estimates)-1):
        for col in range(len(all_estimates[row])):
            if col == row+1:
                plt.text(col,row,line[col], weight='bold')
            elif col > row+1:
                plt.text(col,row,line[col], weight='bold', color='#FFFFFF')
            else:
                plt.text(col,row,line[col], weight='bold', color=cmap(all_estimates[row][col]))

    plt.axis([-0.5, line_length-0.5, -0.5, len(all_estimates)-1])
    plt.gca().invert_yaxis()
    plt.show()
    import pdb; pdb.set_trace()
    plt.close()
