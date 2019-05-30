import torch
import torch.nn as nn

from load_model import load_model
from load_data import load_data

import sklearn.linear_model as sk
import numpy as np

import data
from tqdm import tqdm

import argparse

import os


parser = argparse.ArgumentParser(description='Suprisal plot generator')
parser.add_argument('--tree_data', type=str, default='dahaene_dataset/filtered_sentences.txt',
                    help='path file for the data (in the structure of the Dahaene dataset)')
parser.add_argument('--model_path', type=str, default='PTB.pt.all.e500',
                    help='path file for the saved model (in the structure of awd-lstm)')
parser.add_argument('--training_data', type=str, default='penn',
                    help='path file for the data the model was trained on (in the structure of awd-lstm)')
parser.add_argument('--save_folder', type=str, default='generated_files',
                    help='path file for the hidden states the model generates')
parser.add_argument('--hidden_location', type=str, default='hidden_dictionary.npy',
                    help='path file for the hidden states the model generates')
parser.add_argument('--cell_location', type=str, default='cell_dictionary.npy',
                    help='path file for the cell states the model generates')
parser.add_argument('--targets_location', type=str, default='targets.npy',
                    help='path file for the targets the model generates')
parser.add_argument('--load_data', type=bool, default=False,
                    help='flag to load data')
parser.add_argument('--cross_validation', type=int, default=10,
                    help='number of cross validation folds')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
# loading useful data
print('\nLOADING CORPUS')
model = load_model(args.model_path)
sentences,labels = load_data(args.tree_data)
corpus = data.Corpus(args.training_data)

data_load_failed = False
if args.load_data:
    try:
        hidden_states = np.load(args.hidden_location).item()
        cell_states = np.load(args.cell_location).item()
        targets = np.load(args.targets_location)
    except:
        print('DATA DOES NOT EXIST')
        print('GENERATING DATA')
        data_load_failed = True

if not args.load_data or data_load_failed:
    # parsing dataset
    print('\nPARSING DATASET')
    cell_states = {}
    hidden_states = {}
    targets = []
    for idx in tqdm(range(len(sentences))):
        sentence = sentences[idx]
        label = labels[idx]

        tokenized_data = corpus.safe_tokenize_sentence(sentence.strip())

        ntokens = corpus.dictionary.__len__()

        hidden = model.init_hidden(1)
        input = torch.randint(ntokens, (1, 1), dtype=torch.long).to('cpu')


        for tidx, token in enumerate(tokenized_data[:len(tokenized_data)-1]):

            input.fill_(token.item())
            output, hidden = model(input,hidden)
            for lyr in range(len(hidden)):
                if lyr not in cell_states:
                    cell_states[lyr] = []
                if lyr not in hidden_states:
                    hidden_states[lyr] = []
                h_state,c_state = hidden[lyr]
                cell_states[lyr].append(c_state.detach().numpy())
                hidden_states[lyr].append(h_state.detach().numpy())
            targets.append(label[tidx])

    for key in hidden_states:
        hidden_states[key] = np.array(hidden_states[key]).reshape(len(hidden_states[key]),-1)

    for key in cell_states:
        cell_states[key] = np.array(cell_states[key]).reshape(len(cell_states[key]),-1)

    targets = np.array(targets)

    # save
    np.save(args.save_folder+'/'+args.hidden_location,hidden_states)
    np.save(args.save_folder+'/'+args.cell_location,hidden_states)
    np.save(args.save_folder+'/'+args.targets_location,np.array(targets))

# regressors
def ridge_regression(cross_idx, cross_validation, X, y, all_indices):
    train_indices = np.concatenate((all_indices[0:int(cross_idx/cross_validation * len(X))],all_indices[int((cross_idx+1)/cross_validation * len(X)):]))

    test_indices = all_indices[int(cross_idx/cross_validation * len(X)):int((cross_idx+1)/cross_validation * len(X))]

    train_x = X[train_indices]
    train_y =y[train_indices]
    reg = sk.Ridge().fit(train_x,train_y)

    test_x = X[test_indices]
    test_y = y[test_indices]
    return reg, reg.score(test_x,test_y)

for lyr in hidden_states.keys():
    best_r2_reg = None
    best_r2 = 0

    all_indices = np.arange(len(hidden_states[lyr]))
    np.random.shuffle(all_indices)

    for cidx in range(args.cross_validation):
        reg,r2 = ridge_regression(cidx,args.cross_validation,hidden_states[lyr],targets,all_indices)

        if r2 > best_r2:
            best_r2 = r2
            best_r2_reg = reg

    print('HIDDEN LAYER ',lyr)
    print('BEST REGRESSION R^2 ON HOLD OUT :: ',best_r2)

    mean_coef = best_r2_reg.coef_.mean()
    std_coef = best_r2_reg.coef_.std()

    print('STATISTICALLY SIGNIFICANT UNITS')
    print(np.where(np.abs(best_r2_reg.coef_) >= mean_coef + 3*std_coef )[0])


for lyr in cell_states.keys():
    best_r2_cell = None
    best_r2 = 0

    all_indices = np.arange(len(cell_states[lyr]))
    np.random.shuffle(all_indices)

    for cidx in range(args.cross_validation):
        reg,r2 = ridge_regression(cidx,args.cross_validation,cell_states[lyr],targets,all_indices)

        if r2 > best_r2:
            best_r2 = r2
            best_r2_cell = reg

    print('CELL LAYER ',lyr)
    print('BEST REGRESSION R^2 ON HOLD OUT :: ',best_r2)

    mean_coef = best_r2_cell.coef_.mean()
    std_coef = best_r2_cell.coef_.std()

    print('STATISTICALLY SIGNIFICANT UNITS')
    print(np.where(np.abs(best_r2_cell.coef_) >= mean_coef + 3*std_coef )[0])









# END OF FILE
