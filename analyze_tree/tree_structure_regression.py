import torch
import torch.nn as nn

from load_model import load_model
from load_data import load_data

import sklearn.linear_model as sk
import numpy as np

import data
from tqdm import tqdm

import scipy.stats as stats

import matplotlib.pyplot as plt

import argparse

import utils

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
parser.add_argument('--shuffle_targets', type=bool, default=False,
                    help='flag denoting whether or not target values should be shuffled independentally of the cell states')
parser.add_argument('--gated_forward', type=bool, default=False,
                    help='flag denoting whether or not to use the built in model forward or the forward that we reimplemented to get the gates (setting this as true will analyze gate statistics)')
parser.add_argument('--cross_validation', type=int, default=10,
                    help='number of cross validation folds')
parser.add_argument('--seed', type=int, default=1111,
                    help='torch seed for randomization')
args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)


if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

if not os.path.exists(args.save_folder+'/imgs'):
    os.makedirs(args.save_folder+'/imgs')
# loading useful data
print('\nLOADING CORPUS')
model = load_model(args.model_path)
sentences,labels = load_data(args.tree_data,'open_nodes')
corpus = data.Corpus(args.training_data)


if args.gated_forward:
    print('USING GATED FORWARD')
    model_values = utils.get_model_values(model)

data_load_failed = False
if args.load_data:
    try:
        print('LOADING DATA')
        hidden_states = np.load(args.hidden_location).item()
        cell_states = np.load(args.cell_location).item()
        targets = np.load(args.targets_location)

        if args.gated_forward:
            forget_gates = np.load(args.save_folder+'/forget_gates.npy').item()
            input_gates = np.load(args.save_folder+'/input_gates.npy').item()
            output_gates = np.load(args.save_folder+'/output_gates.npy').item()

    except:
        print('DATA DOES NOT EXIST')
        print('GENERATING DATA')
        data_load_failed = True

if not args.load_data or data_load_failed:
    # parsing dataset
    print('\nPARSING DATASET')
    cell_states = {}
    hidden_states = {}
    if args.gated_forward:
        forget_gates = {}
        output_gates = {}
        input_gates = {}

    targets = []
    for idx in tqdm(range(len(sentences))):
        sentence = sentences[idx]
        label = labels[idx]

        tokenized_data = corpus.safe_tokenize_sentence(sentence.strip())

        ntokens = corpus.dictionary.__len__()

        hidden = model.init_hidden(1)
        input = torch.randint(ntokens, (1, 1), dtype=torch.long).to('cpu')


        for tidx, token in enumerate(tokenized_data[:len(tokenized_data)-1]):

            if not args.gated_forward:
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
            else:
                input.fill_(token.item())
                output, hidden, out_gates, raw_outputs, outputs = utils.gated_forward(model, model_values, input, hidden)


                for lyr in range(len(hidden)):
                    if lyr not in cell_states:
                        cell_states[lyr] = []
                    if lyr not in hidden_states:
                        hidden_states[lyr] = []

                    if lyr not in forget_gates:
                        forget_gates[lyr] = {}
                    if lyr not in output_gates:
                        output_gates[lyr] = {}
                    if lyr not in input_gates:
                        input_gates[lyr] = {}

                    if idx not in forget_gates[lyr]:
                        forget_gates[lyr][idx] = []
                    if idx not in output_gates[lyr]:
                        output_gates[lyr][idx] = []
                    if idx not in input_gates[lyr]:
                        input_gates[lyr][idx] = []



                    f_gates,i_gates,o_gates,g_gates = out_gates[lyr]


                    h_state,c_state = hidden[lyr]
                    cell_states[lyr].append(c_state.detach().numpy())
                    hidden_states[lyr].append(h_state.detach().numpy())

                    forget_gates[lyr][idx].append(f_gates[0].detach().numpy())
                    output_gates[lyr][idx].append(o_gates[0].detach().numpy())
                    input_gates[lyr][idx].append(i_gates[0].detach().numpy())

                targets.append(label[tidx])

    for key in hidden_states:
        hidden_states[key] = np.array(hidden_states[key]).reshape(len(hidden_states[key]),-1)

    for key in cell_states:
        cell_states[key] = np.array(cell_states[key]).reshape(len(cell_states[key]),-1)

    if args.gated_forward:
        for key in forget_gates:
            for sidx in forget_gates[key]:
                forget_gates[key][sidx] = np.array(forget_gates[key][sidx]).reshape(len(forget_gates[key][sidx]),-1)
        for key in output_gates:
            for sidx in output_gates[key]:
                output_gates[key][sidx] = np.array(output_gates[key][sidx]).reshape(len(output_gates[key][sidx]),-1)
        for key in input_gates:
            for sidx in input_gates[key]:
                input_gates[key][sidx] = np.array(input_gates[key][sidx]).reshape(len(input_gates[key][sidx]),-1)

    targets = np.array(targets)

    # save
    np.save(args.save_folder+'/'+args.hidden_location,hidden_states)
    np.save(args.save_folder+'/'+args.cell_location,cell_states)
    np.save(args.save_folder+'/'+args.targets_location,targets)
    if args.gated_forward:
        np.save(args.save_folder+'/forget_gates',forget_gates)
        np.save(args.save_folder+'/input_gates',input_gates)
        np.save(args.save_folder+'/output_gates',output_gates)


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

if args.shuffle_targets:
    print('SHUFFLING TARGETS')
    np.random.shuffle(targets)

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
    if best_r2_reg is not None:
        print('HIDDEN LAYER ',lyr)
        print('BEST REGRESSION R^2 ON HOLD OUT :: ',best_r2)

        mean_coef = best_r2_reg.coef_.mean()
        std_coef = best_r2_reg.coef_.std()

        print('STATISTICALLY SIGNIFICANT UNITS')
        print(np.where(np.abs(best_r2_reg.coef_) >= mean_coef + 3*std_coef )[0])
    else:
        print('UNABLE TO REGRESS')


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
    if best_r2_cell is not None:
        print('CELL LAYER ',lyr)
        print('BEST REGRESSION R^2 ON HOLD OUT :: ',best_r2)

        mean_coef = best_r2_cell.coef_.mean()
        std_coef = best_r2_cell.coef_.std()

        print('STATISTICALLY SIGNIFICANT UNITS')
        print(np.where(np.abs(best_r2_cell.coef_) >= mean_coef + 3*std_coef )[0])
    else:
        print('UNABLE TO REGRESS')

if args.gated_forward:
# Differece GAS
    for lyr in range(len(forget_gates)):
        print('\n\n\nLAYER ',lyr+1)

        # Forget Gates

        layer_gates = forget_gates[lyr]

        merge_values_pre = []
        other_values_pre = []
        merge_values_post = []
        other_values_post = []

        for sidx in range(len(sentences)):
            sentence = sentences[sidx]
            pre_label = labels[sidx][:len(labels[sidx])-1]
            post_label = labels[sidx][1:]

            gates = layer_gates[sidx].mean(1)
            diffs = gates[1:] - gates[:len(gates)-1]

            # depth_change_locs = []

            # Are merge values statistically significant than the other values?
            pre_l = -1
            for lidx,l in enumerate(pre_label[:len(pre_label)-1]):
                if l < pre_l:
                    # MERGE
                    merge_values_pre.append(diffs[lidx])
                    depth_change_locs.append(lidx)
                else:
                    other_values_pre.append(diffs[lidx])
                pre_l = l
            #
            # for x_line in depth_change_locs:
            #     plt.axvline(x=x_line)
            # plt.plot(diffs)
            # plt.xticks(range(len(diffs)),sentence.split(' ')[:len(labels[sidx])-1])
            # plt.show()
            #
            # import pdb; pdb.set_trace()


            pre_l = -1
            for lidx,l in enumerate(post_label[:len(post_label)-1]):
                if l < pre_l:
                    # MERGE
                    merge_values_post.append(diffs[lidx])
                else:
                    other_values_post.append(diffs[lidx])
                pre_l = l
            merge_values_post.append(diffs[len(diffs)-1])


        merge_values_pre = np.array(merge_values_pre)
        other_values_pre = np.array(other_values_pre)

        merge_values_post = np.array(merge_values_post)
        other_values_post = np.array(other_values_post)

        ks, p_val = stats.ks_2samp(other_values_pre, merge_values_pre)
        print('\nPRE LABELLING')
        print('Kolmogorov-Smirnoff p-value::',p_val)
        ts, p_val = stats.ttest_1samp(merge_values_pre,other_values_pre.mean())
        print('T-Test p-value::',p_val)

        plt.title('Layer '+str(lyr+1)+':\n Comparing Distribution of Merge (Pre Labelling) vs Other Forget Statistic\n(p-value = '+str(round(p_val,5))+')')
        plt.ylabel('Frequency')
        plt.xlabel('Statistic Value')
        plt.hist(other_values_pre,bins=25,label='Other Values')
        plt.hist(merge_values_pre,bins=25,label='Merge Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.save_folder+'/imgs/forget_plots_pre_'+str(lyr+1)+'.png')
        plt.close()




        ks, p_val = stats.ks_2samp(other_values_post, merge_values_post)
        print('\nPOST LABELLING')
        print('Kolmogorov-Smirnoff p-value::',p_val)
        ts, p_val = stats.ttest_1samp(merge_values_post,other_values_post.mean())
        print('T-Test p-value::',p_val)

        plt.title('Layer '+str(lyr+1)+':\n Comparing Distribution of Merge (Post Labelling) vs Other Forget Statistic\n(p-value = '+str(round(p_val,5))+')')
        plt.ylabel('Frequency')
        plt.xlabel('Statistic Value')
        plt.hist(other_values_post,bins=25,label='Other Values')
        plt.hist(merge_values_post,bins=25,label='Merge Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.save_folder+'/imgs/forget_plots_post_'+str(lyr+1)+'.png')
        plt.close()












# END OF FILE
