import torch.nn as nn
import torch

import Gated_LSTM

import numpy as np
import random

from tqdm import tqdm

import matplotlib.pyplot as plt

import utils

dataset_file = 'datasets/a2bcd_all_training.txt'
rules_file = 'language_rules/a2bcd_all.rul'

num_layers = 1
hidden_size = 3
num_epochs = 5
PATH = 'models/model_a2bcd_all_l_'+str(num_layers)+'_h_'+str(hidden_size)+'_ep_'+str(num_epochs)

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

train_dataset = []
with open(dataset_file) as data:
    for line in data:
        train_dataset.append(line.strip())


model = Gated_LSTM.GatedLSTM(input_size,hidden_size)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

all_loss = []
accuracy = []
for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0
    total_predictions = 0
    total_mistakes = 0
    random.shuffle(train_dataset)
    for line_index, line in enumerate(train_dataset):
        optimizer.zero_grad()
        # print(str(line_index) +' of '+str(len(train_dataset)))
        line_length = len(line)
        x = []
        labels = []
        for i in range(line_length):
            current, character_idx = utils.character_to_vector(line[i], vocabulary)
            x.append(current)
            labels.append(character_idx)

        labels = torch.Tensor(labels[1:]).long()
        x = torch.Tensor(x).reshape(1,line_length,-1)
        output, hidden, _ = model(x,None)

        decoded = model.decoder(output[:,:line_length-1,:])
        predictions = nn.functional.log_softmax(decoded[0]).reshape(line_length-1,-1)
        probabilities = torch.exp(predictions)

        classes = torch.multinomial(probabilities,1,replacement=True).reshape(-1)
        mistakes = len(torch.nonzero(classes - labels))

        total_mistakes += mistakes
        total_predictions += (line_length-1)

        loss = criterion(predictions, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    all_loss.append(epoch_loss)
    accuracy.append((total_predictions-total_mistakes)/total_predictions)

plt.title('NLLLoss Over Epochs')
plt.plot(range(num_epochs),all_loss)
plt.xlabel('Epoch')
plt.ylabel('NLLLoss')
plt.show()
plt.close()
plt.title('Accuracy Over Epochs')
plt.plot(range(num_epochs),accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
plt.close()

torch.save(model.state_dict(), PATH)
