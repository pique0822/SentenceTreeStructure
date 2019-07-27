import torch.nn as nn
import torch

import Gated_LSTM

import numpy as np
import random

from tqdm import tqdm

import matplotlib.pyplot as plt

import utils

dataset_file = 'datasets/a2bcd_testing.txt'
rules_file = 'language_rules/a2bcd.rul'
num_layers = 1
hidden_size = 3
PATH = 'models/model_a2bcd_l_1_h_3_ep_10'

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
with open(dataset_file) as data:
    for line in data:
        test_dataset.append(line.strip())



model = Gated_LSTM.GatedLSTM(input_size,hidden_size)
model.load_state_dict(torch.load(PATH))
model.eval()



total_predictions = 0
total_mistakes = 0

follow_up = {}
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
    output, hidden, _ = model(x,None)

    decoded = model.decoder(output[:,:line_length-1,:])
    predictions = nn.functional.log_softmax(decoded[0]).reshape(line_length-1,-1)
    probabilities = torch.exp(predictions)
    classes = torch.multinomial(probabilities,1,replacement=True).reshape(-1)
    # classes = predictions.argmax(1)
    mistakes = len(torch.nonzero(classes - labels))

    for i in range(len(inputs)):
        inp = vocabulary[inputs[i]]
        out = classes[i]

        if inp not in follow_up:
            follow_up[inp] = [0]*len(vocabulary)

        follow_up[inp][out] += 1

    total_mistakes += mistakes
    total_predictions += (line_length-1)

print('Test Accuracy', (total_predictions-total_mistakes)/total_predictions)

for inp in follow_up:
    plt.bar(range(len(vocabulary)),follow_up[inp])
    plt.xticks(range(len(vocabulary)),vocabulary)
    plt.xlabel('Next Character')
    plt.ylabel('Frequency')
    plt.title('Distribution of Next Character from '+str(inp))
    plt.show()
    plt.close()
