import torch
import torch.nn as nn

from load_model import load_model
from load_data import load_data

import sklearn.linear_model as sk
import sklearn.metrics as skm
import sklearn.decomposition as skd

import numpy as np

import pandas as pd

import data
from tqdm import tqdm

import scipy.stats as stats

import matplotlib.pyplot as plt

import argparse

import utils

import os

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

model_path = 'PTB.pt.all.e500'
tree_data = 'dahaene_dataset/filtered_sentences.txt'
training_data = 'penn'

model = load_model(model_path)
sentences,labels = load_data(tree_data,'open_nodes')
corpus = data.Corpus(training_data)

model_values = utils.get_model_values(model)

for idx in range(len(sentences)):
    sentence = sentences[idx]
    label = labels[idx]

    tokenized_data = corpus.safe_tokenize_sentence(sentence.strip())

    ntokens = corpus.dictionary.__len__()

    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to('cpu')


    previous_label = -1
    for tidx, token in enumerate(tokenized_data[:len(tokenized_data)-1]):

        input.fill_(token.item())
        output, hidden, out_gates, raw_outputs, outputs = utils.gated_forward(model, model_values, input, hidden)

        forget_gates = []
        input_gates = []
        output_gates = []
        c_tilde_gates = []

        hidden_states = []
        cell_states = []
        # import pdb; pdb.set_trace()
        for layer in range(len(hidden)):
            h,c = hidden[layer]

            hidden_states.append(h)
            cell_states.append(c)

            (f_gates,i_gates,o_gates,g_gates) = out_gates[layer]
            forget_gates.append(f_gates)
            input_gates.append(i_gates)
            output_gates.append(o_gates)
            c_tilde_gates.append(g_gates)

        utils.back_prop_relevance(model, model_values, forget_gates, input_gates, output_gates, c_tilde_gates, hidden_states, cell_states)
        import pdb; pdb.set_trace()










#EOF
