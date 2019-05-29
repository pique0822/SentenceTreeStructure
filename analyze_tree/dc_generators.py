import torch
from torch import nn
from torch.nn import functional as F

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

import scipy.io

import data
import model

from utils import batchify, get_batch, repackage_hidden

import matplotlib.pyplot as plt

import pdb

import sklearn.linear_model as sk
import numpy as np

model_path = '../PTB.pt.all.e500'



# Load the pretrained model
with open(model_path, 'rb') as f:
    model, criterion, optimizer = torch.load(f, map_location='cpu')

print('=== MODEL INFORMATION ===')
print(model)


print('\n\n=== MODEL PARAMATERS ===')
params = {}
for name, param in model.named_parameters():
    print(name)
    params[name] = param

model_values = {}
### LAYER 1 ###
# Current Timestep

encoder_weights = params['encoder.weight']
decoder_bias = params['decoder.bias']

for layer in range(len(model.rnns)):

    hidden_size = model.rnns[layer].module.hidden_size
    name = 'rnns.'+str(layer)+'.module.'

    w_ii_l0 = params[name+'weight_ih_l0'][0:hidden_size]
    w_if_l0 = params[name+'weight_ih_l0'][hidden_size:2*hidden_size]
    w_ig_l0 = params[name+'weight_ih_l0'][2*hidden_size:3*hidden_size]
    w_io_l0 = params[name+'weight_ih_l0'][3*hidden_size:4*hidden_size]

    b_ii_l0 = params[name+'bias_ih_l0'][0:hidden_size]
    b_if_l0 = params[name+'bias_ih_l0'][hidden_size:2*hidden_size]
    b_ig_l0 = params[name+'bias_ih_l0'][2*hidden_size:3*hidden_size]
    b_io_l0 = params[name+'bias_ih_l0'][3*hidden_size:4*hidden_size]

    input_vals = (w_ii_l0,b_ii_l0,w_if_l0,b_if_l0,w_ig_l0,b_ig_l0,w_io_l0,b_io_l0)
    # Recurrent
    w_hi_l0 = params[name+'weight_hh_l0_raw'][0:hidden_size]
    w_hf_l0 = params[name+'weight_hh_l0_raw'][hidden_size:2*hidden_size]
    w_hg_l0 = params[name+'weight_hh_l0_raw'][2*hidden_size:3*hidden_size]
    w_ho_l0 = params[name+'weight_hh_l0_raw'][3*hidden_size:4*hidden_size]

    b_hi_l0 = params[name+'bias_hh_l0'][0:hidden_size]
    b_hf_l0 = params[name+'bias_hh_l0'][hidden_size:2*hidden_size]
    b_hg_l0 = params[name+'bias_hh_l0'][2*hidden_size:3*hidden_size]
    b_ho_l0 = params[name+'bias_hh_l0'][3*hidden_size:4*hidden_size]

    hidden_vals = (w_hi_l0,b_hi_l0,w_hf_l0,b_hf_l0,w_hg_l0,b_hg_l0,w_ho_l0,b_ho_l0)

    model_values[layer] = (input_vals,hidden_vals)


#
def gated_forward(input, hidden, return_h=False,return_gates=False):
    emb = embedded_dropout(model.encoder, input, dropout=model.dropoute if model.training else 0)

    emb = model.lockdrop(emb, model.dropouti)

    raw_output = emb
    new_hidden = []
    #raw_output, hidden = model.rnn(emb, hidden)
    raw_outputs = []
    outputs = []

    out_gates = []

    for l, rnn in enumerate(model.rnns):
        # print('LAYER ',l)
        (h0_l0, c0_l0) = hidden[l]
        i_vals, h_vals = model_values[l]

        (w_ii_l0,b_ii_l0,w_if_l0,b_if_l0,w_ig_l0,b_ig_l0,w_io_l0,b_io_l0) = i_vals

        (w_hi_l0,b_hi_l0,w_hf_l0,b_hf_l0,w_hg_l0,b_hg_l0,w_ho_l0,b_ho_l0) = h_vals

        gated_out = []

        f_gates, i_gates, o_gates, g_gates = [],[],[],[]
        for seq_i in range(len(raw_output)):
            inp = raw_output[seq_i]

            # forget gate
            f_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_if_l0)) + b_if_l0) + (torch.matmul(h0_l0,torch.t(w_hf_l0)) + b_hf_l0))


            # input gate
            i_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_ii_l0)) + b_ii_l0) + (torch.matmul(h0_l0,torch.t(w_hi_l0)) + b_hi_l0))

            # output gate
            o_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_io_l0)) + b_io_l0) + (torch.matmul(h0_l0, torch.t(w_ho_l0)) + b_ho_l0))


            # intermediate cell state
            c_tilde_l0 = torch.tanh((torch.matmul(inp,torch.t(w_ig_l0)) + b_ig_l0) + (torch.matmul(h0_l0, torch.t(w_hg_l0)) + b_hg_l0))

            # current cell state
            c0_l0 = f_g_l0 * c0_l0 + i_g_l0 * c_tilde_l0

            # hidden state
            h0_l0 = o_g_l0 * torch.tanh(c0_l0)

            new_h = (h0_l0,c0_l0)
            gated_out.append(h0_l0)

            f_gates.append(f_g_l0)
            i_gates.append(i_g_l0)
            o_gates.append(o_g_l0)
            g_gates.append(c0_l0)

        gates = (f_gates,i_gates,o_gates,g_gates)
        out_gates.append(gates)
        # pdb.set_trace()
        out = torch.stack(gated_out).reshape(len(gated_out),h0_l0.shape[1],h0_l0.shape[2])
        raw_output = out

        new_hidden.append(new_h)
        raw_outputs.append(raw_output)
        if l != model.nlayers - 1:
            #model.hdrop(raw_output)
            raw_output = model.lockdrop(raw_output, model.dropouth)
            outputs.append(raw_output)
    hidden = new_hidden

    output = model.lockdrop(raw_output, model.dropout)
    outputs.append(output)

    result = output.view(output.size(0)*output.size(1), output.size(2))

    if return_h and return_gates:
        return result, hidden, out_gates, raw_outputs, outputs
    elif return_h:
        return result, hidden, raw_outputs, outputs
    elif return_gates:
        return result, hidden, out_gates
    return result, hidden

def batchify(data, bsz, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, seq_len=1):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source,bsz):
    # Turn on evaluation mode which disables dropout.

    model.eval()

    total_loss = 0
    ntokens = len(corpus.dictionary)

    hidden = model.init_hidden(bsz)

    data, targets = data_source[0:len(data_source)-1],data_source[1:]

    result, hidden, out_gates, raw_outputs, outputs = gated_forward(data, hidden, return_gates=True, return_h=True)
    return out_gates, outputs

# used to parse line (should change based on dataset format)
def parse_line(line):
    sentence, tree_struct, open_nodes, adj_nodes = line.split('|')
    rel = open_nodes.lstrip().strip().replace("\\","")
    rel = rel.replace("n","")
    rel = rel.replace("'","")
    brackets_open = [int(x) for x in rel.split()]
    return sentence.lstrip().strip(), brackets_open

data_path = "/Users/DelRio/Desktop/GeometricNeuroAI/awd-lstm-lm/data/penn/"
input_path = "../data_generation/tree_structure/filtered_sentences.txt"

print('\n=== DEFINING CORPUS ===')
corpus = data.Corpus(data_path)

print('\n=== TESTING MODEL ===')

print(input_path+'\n')
forget_gates = {}
input_gates = {}
output_gates = {}
cell_states = {}
hidden_states = {}

relevant_labels = {}

with open(input_path) as input_file:
    for line in input_file:
        sentence, labels = parse_line(line)

        tokenized_data = corpus.safe_tokenize_sentence(sentence.strip())

        batch_size = 1
        input_data = batchify(tokenized_data,batch_size,False)
        gate_data, outputs = evaluate(input_data,batch_size)

        for lyr, gates in enumerate(gate_data):
            if lyr not in forget_gates:
                forget_gates[lyr] = []
            if lyr not in input_gates:
                input_gates[lyr] = []
            if lyr not in output_gates:
                output_gates[lyr] = []
            if lyr not in cell_states:
                cell_states[lyr] = []
            if lyr not in hidden_states:
                hidden_states[lyr] = []
            if lyr not in relevant_labels:
                relevant_labels[lyr] = []

            f_g_l0,i_g_l0,o_g_l0,c_tilde_l0 = gates
            hidden_values = outputs[lyr]
            for word in range(len(f_g_l0)):
                forget_gates[lyr].append(f_g_l0[word].detach().numpy())
                input_gates[lyr].append(i_g_l0[word].detach().numpy())
                output_gates[lyr].append(o_g_l0[word].detach().numpy())
                cell_states[lyr].append(c_tilde_l0[word].detach().numpy())
                hidden_states[lyr].append(hidden_values[word].detach().numpy())
            relevant_labels[lyr].extend(labels)
            # uncomment to only see the first layer for testing
            # break

train_percent = 0.1
letter_to_name = {'f':'Forget','o':'Output','i':'Input','c':'Cell State','h':'Hidden State'}
print('\n=== PLOTTING VIZ ===')
for lyr in range(len(model.rnns)):
    print('\n\n\nLayer '+str(lyr) + '\n\n\n')
    relevant_gates = []
    for letter in ['f','o','i','c','h']:
        if letter == 'f':
            relevant_gates = forget_gates[lyr]
        elif letter == 'o':
            relevant_gates = output_gates[lyr]
        elif letter == 'i':
            relevant_gates = input_gates[lyr]
        elif letter == 'c':
            relevant_gates = cell_states[lyr]
        elif letter == 'h':
            relevant_gates = hidden_states[lyr]

        print('\n'+letter_to_name[letter])
        train_number = int(train_percent*len(relevant_gates))

        # correct
        training_indices = np.random.choice(range(len(relevant_gates)),train_number,replace=False)
        test_indices = list(set(range(len(relevant_gates))).difference(training_indices))

        fgates = np.array(relevant_gates)
        fgates = fgates[training_indices]
        fgates = fgates.reshape(train_number,-1)

        labels = np.array(relevant_labels[lyr])
        labels = labels[training_indices]


        linreg = sk.LinearRegression().fit(fgates,labels)
        l2_penalty = sk.Ridge().fit(fgates,labels)
        l1_penalty = sk.Lasso().fit(fgates,labels)

        test_fgates = np.array(relevant_gates)
        test_fgates = test_fgates[test_indices]
        test_fgates = test_fgates.reshape(len(test_fgates),-1)

        test_labels = np.array(relevant_labels[lyr])
        test_labels = test_labels[test_indices]

        lr_score = linreg.score(test_fgates,test_labels)
        l2_score = l2_penalty.score(test_fgates,test_labels)
        l1_score = l1_penalty.score(test_fgates,test_labels)

        print('LinearRegressor Score ::',lr_score)
        print('L2 Regressor Score    ::',l2_score)
        print('L1 Regressor Score    ::',l1_score)

        # finding the biggest outlier values
        l2_weights = l2_penalty.coef_
        l2_w_mean = l2_weights.mean()
        l2_w_std = (l2_weights.var())**.5
        print('Outlier L2 Weight Indices')
        l2_outliers = np.abs(l2_weights) > l2_w_mean + 3*l2_w_std
        l2_indices = np.where(l2_outliers == True)
        print(l2_indices)


# END OF FILE
