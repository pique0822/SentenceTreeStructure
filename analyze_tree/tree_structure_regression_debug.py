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

import seaborn as sns
pca_analysis = False
significance_analysis = False
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
parser.add_argument('--depth_targets_location', type=str, default='depth_targets.npy',
                    help='path file for the depth targets the model generates')
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

print("\n\n\n\n\n\n\nDEBUG FILE\n\n\n\n\n\n\n")


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
        print('HIDDEN STATES')
        cell_states = np.load(args.cell_location).item()
        print('CELL STATES')
        targets = np.load(args.targets_location)

        depth_targets = np.load(args.depth_targets_location)
        print('TARGETS')


        # other_hidden_states = np.load('generated_files/hidden_dictionary.npy')

        if args.gated_forward:
            forget_gates = np.load(args.save_folder+'/forget_gates.npy').item()
            print('FORGET GATES')
            input_gates = np.load(args.save_folder+'/input_gates.npy').item()
            print('INPUT GATES')
            output_gates = np.load(args.save_folder+'/output_gates.npy').item()
            print('OUTPUT GATES')
            hidden_gates = np.load(args.save_folder+'/hidden_gates.npy').item()
            print('HIDDEN GATES')
            cell_gates = np.load(args.save_folder+'/cell_gates.npy').item()
            print('CELL GATES')

    except:
        print('DATA DOES NOT EXIST')
        print('GENERATING DATA')
        data_load_failed = True

if not args.load_data or data_load_failed:
    # parsing dataset
    print('\nPARSING DATASET')
    cell_states = {}
    hidden_states = {}

    merged_hidden_states_eos = {}
    merged_hidden_states_mid = {}
    other_hidden_states = {}

    if args.gated_forward:
        forget_gates = {}
        output_gates = {}
        input_gates = {}

        cell_gates = {}
        hidden_gates = {}

    targets = []
    depth_targets = [] # Simply measures the distance to the beginning of the sentence
    for idx in tqdm(range(len(sentences))):
        sentence = sentences[idx]
        label = labels[idx]

        depth = list(range(1,len(label)+1))

        tokenized_data = corpus.safe_tokenize_sentence(sentence.strip())

        ntokens = corpus.dictionary.__len__()

        hidden = model.init_hidden(1)
        input = torch.randint(ntokens, (1, 1), dtype=torch.long).to('cpu')


        previous_label = -1
        for tidx, token in enumerate(tokenized_data[:len(tokenized_data)-1]):

            if not args.gated_forward:
                input.fill_(token.item())
                output, hidden = model(input,hidden)
                for lyr in range(len(hidden)):
                    if lyr not in cell_states:
                        cell_states[lyr] = []
                    if lyr not in hidden_states:
                        hidden_states[lyr] = []

                    if lyr not in merged_hidden_states_eos:
                        merged_hidden_states_eos[lyr] = []
                    if lyr not in merged_hidden_states_mid:
                        merged_hidden_states_mid[lyr] = []
                    if lyr not in other_hidden_states:
                        other_hidden_states[lyr] = []

                    h_state,c_state = hidden[lyr]
                    cell_states[lyr].append(c_state.detach().numpy())
                    hidden_states[lyr].append(h_state.detach().numpy())

                    if label[tidx] <= previous_label:
                        # merge
                        merged_hidden_states_mid[lyr].append(h_state.detach().numpy())
                    else:
                        other_hidden_states[lyr].append(h_state.detach().numpy())

                    merged_hidden_states_eos[lyr].append(h_state.detach().numpy())

                targets.append(label[tidx])
                depth_targets.append(depth[tidx])

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
                    if lyr not in cell_gates:
                        cell_gates[lyr] = {}
                    if lyr not in hidden_gates:
                        hidden_gates[lyr] = {}

                    if idx not in forget_gates[lyr]:
                        forget_gates[lyr][idx] = []
                    if idx not in output_gates[lyr]:
                        output_gates[lyr][idx] = []
                    if idx not in input_gates[lyr]:
                        input_gates[lyr][idx] = []
                    if idx not in cell_gates[lyr]:
                        cell_gates[lyr][idx] = []
                    if idx not in hidden_gates[lyr]:
                        hidden_gates[lyr][idx] = []

                    if lyr not in merged_hidden_states_eos:
                        merged_hidden_states_eos[lyr] = []
                    if lyr not in merged_hidden_states_mid:
                        merged_hidden_states_mid[lyr] = []
                    if lyr not in other_hidden_states:
                        other_hidden_states[lyr] = []



                    f_gates,i_gates,o_gates,g_gates = out_gates[lyr]


                    h_state,c_state = hidden[lyr]
                    cell_states[lyr].append(c_state.detach().numpy())
                    hidden_states[lyr].append(h_state.detach().numpy())

                    cell_gates[lyr][idx].append(c_state.detach().numpy())
                    hidden_gates[lyr][idx].append(h_state.detach().numpy())

                    forget_gates[lyr][idx].append(f_gates[0].detach().numpy())
                    output_gates[lyr][idx].append(o_gates[0].detach().numpy())
                    input_gates[lyr][idx].append(i_gates[0].detach().numpy())

                    if label[tidx] <= previous_label:
                        # merge
                        merged_hidden_states_mid[lyr].append(h_state.detach().numpy())
                    else:
                        other_hidden_states[lyr].append(h_state.detach().numpy())


                    merged_hidden_states_eos[lyr].append(h_state.detach().numpy())

                targets.append(label[tidx])
                depth_targets.append(depth[tidx])

                previous_label = label[tidx]



    for key in hidden_states:
        hidden_states[key] = np.array(hidden_states[key]).reshape(len(hidden_states[key]),-1)

    for key in cell_states:
        cell_states[key] = np.array(cell_states[key]).reshape(len(cell_states[key]),-1)

    for key in merged_hidden_states_mid:
        if len(merged_hidden_states_mid[key]) != 0:
            merged_hidden_states_mid[key] = np.array(merged_hidden_states_mid[key]).reshape(len(merged_hidden_states_mid[key]),-1)

    for key in merged_hidden_states_eos:
        if len(merged_hidden_states_eos[key]) != 0:
            merged_hidden_states_eos[key] = np.array(merged_hidden_states_eos[key]).reshape(len(merged_hidden_states_eos[key]),-1)

    for key in other_hidden_states:
        if len(other_hidden_states[key]) != 0:
            other_hidden_states[key] = np.array(other_hidden_states[key]).reshape(len(other_hidden_states[key]),-1)

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

        for key in cell_gates:
            for sidx in cell_gates[key]:
                cell_gates[key][sidx] = np.array(cell_gates[key][sidx]).reshape(len(cell_gates[key][sidx]),-1)
        for key in hidden_gates:
            for sidx in hidden_gates[key]:
                hidden_gates[key][sidx] = np.array(hidden_gates[key][sidx]).reshape(len(hidden_gates[key][sidx]),-1)

    targets = np.array(targets)
    depth_targets = np.array(depth_targets)

    # save
    np.save(args.save_folder+'/'+args.hidden_location,hidden_states)
    np.save(args.save_folder+'/'+args.cell_location,cell_states)
    np.save(args.save_folder+'/'+args.targets_location,targets)
    np.save(args.save_folder+'/depth_'+args.targets_location,depth_targets)

    np.save(args.save_folder+'/merged_eos_'+args.hidden_location,merged_hidden_states_eos)
    np.save(args.save_folder+'/merged_mid_'+args.hidden_location,merged_hidden_states_mid)
    np.save(args.save_folder+'/other_'+args.hidden_location,other_hidden_states)

    if args.gated_forward:
        np.save(args.save_folder+'/forget_gates',forget_gates)
        np.save(args.save_folder+'/input_gates',input_gates)
        np.save(args.save_folder+'/output_gates',output_gates)
        np.save(args.save_folder+'/cell_gates',cell_gates)
        np.save(args.save_folder+'/hidden_gates',hidden_gates)


# regressors
def ridge_regression(cross_idx, cross_validation, X, y, all_indices):
    train_indices = np.concatenate((all_indices[0:int(cross_idx/cross_validation * len(X))],all_indices[int((cross_idx+1)/cross_validation * len(X)):]))

    test_indices = all_indices[int(cross_idx/cross_validation * len(X)):int((cross_idx+1)/cross_validation * len(X))]

    train_x = X[train_indices]
    train_y =y[train_indices]
    reg = sk.Ridge().fit(train_x,train_y)

    test_x = X[test_indices]
    test_y = y[test_indices]

    pred_y = reg.predict(test_x)

    return reg, reg.score(test_x,test_y), skm.mean_squared_error(test_y,pred_y)

if args.shuffle_targets:
    print('SHUFFLING TARGETS')
    np.random.shuffle(targets)


if significance_analysis:
    # STATISTICAL ANALYSIS
    #type will be defined as following 0 is normal, 1 is ctrl 1 and 2 is ctrl 2
    hidden_regression = {'Layer':[],'R^2':[],'MSE':[],'type':[]}

    # NORMAL
    all_indices = np.arange(len(hidden_states[lyr]))
    np.random.shuffle(all_indices)

    for lyr in hidden_states.keys():
        best_r2_reg = None
        best_r2 = 0

        for cidx in range(args.cross_validation):
            reg,r2,mse = ridge_regression(cidx,args.cross_validation,hidden_states[lyr],targets,all_indices)

            hidden_regression['Layer'].append(lyr)
            hidden_regression['R^2'].append(r2)
            hidden_regression['MSE'].append(mse)
            hidden_regression['type'].append('Normal')


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

    # CONTROL 2 DEPTH TARGETS
    all_indices = np.arange(len(hidden_states[lyr]))
    np.random.shuffle(all_indices)
    for lyr in hidden_states.keys():

        for cidx in range(args.cross_validation):
            reg,r2,mse = ridge_regression(cidx,args.cross_validation,hidden_states[lyr],depth_targets,all_indices)

            hidden_regression['Layer'].append(lyr)
            hidden_regression['R^2'].append(r2)
            hidden_regression['MSE'].append(mse)
            hidden_regression['type'].append('Control 2 (Depth From Beginning of Sentence)')



    ax = sns.lineplot(x="Layer", y="R^2", hue='type', data=hidden_regression)
    plt.title('Hidden State R^2 Score Over Layers')
    plt.xlabel('Layer')
    plt.xticks(range(len(hidden_states.keys())))
    plt.ylabel('R^2 Score')
    plt.legend(loc='upper right')
    plt.savefig('generated_files/imgs/hidden_r2_layers_ctrl2.png')
    plt.close()

    ax = sns.lineplot(x="Layer", y="MSE", hue='type', data=hidden_regression)
    plt.title('Hidden State MSE Over Layers')
    plt.xlabel('Layer')
    plt.xticks(range(len(hidden_states.keys())))
    plt.ylabel('MSE')
    plt.legend(loc='upper right')
    plt.savefig('generated_files/imgs/hidden_mse_layers_ctrl2.png')
    plt.close()

    # CONTROL 1 SHUFFLE TARGETS
    random_shuffled_targets = targets.copy()
    np.random.shuffle(random_shuffled_targets)

    all_indices = np.arange(len(hidden_states[lyr]))
    np.random.shuffle(all_indices)

    for lyr in hidden_states.keys():

        for cidx in range(args.cross_validation):
            reg,r2,mse = ridge_regression(cidx,args.cross_validation,hidden_states[lyr],random_shuffled_targets,all_indices)

            hidden_regression['Layer'].append(lyr)
            hidden_regression['R^2'].append(r2)
            hidden_regression['MSE'].append(mse)
            hidden_regression['type'].append('Control 1 (Randomly Shuffled Targets)')

    ax = sns.lineplot(x="Layer", y="R^2", hue='type', data=hidden_regression)
    plt.title('Hidden State R^2 Score Over Layers')
    plt.xlabel('Layer')
    plt.xticks(range(len(hidden_states.keys())))
    plt.ylabel('R^2 Score')
    plt.legend(loc='upper right')
    plt.savefig('generated_files/imgs/hidden_r2_layers.png')
    plt.close()

    ax = sns.lineplot(x="Layer", y="MSE", hue='type', data=hidden_regression)
    plt.title('Hidden State MSE Over Layers')
    plt.xlabel('Layer')
    plt.xticks(range(len(hidden_states.keys())))
    plt.ylabel('MSE')
    plt.legend(loc='upper right')
    plt.savefig('generated_files/imgs/hidden_mse_layers.png')
    plt.close()


    # CELL STATES

    #type will be defined as following 0 is normal, 1 is ctrl 1 and 2 is ctrl 2
    cell_regression = {'Layer':[],'R^2':[],'MSE':[],'type':[]}

    # NORMAL
    for lyr in cell_states.keys():
        best_r2_reg = None
        best_r2 = 0

        all_indices = np.arange(len(cell_states[lyr]))
        np.random.shuffle(all_indices)

        for cidx in range(args.cross_validation):
            reg,r2,mse = ridge_regression(cidx,args.cross_validation,cell_states[lyr],targets,all_indices)

            cell_regression['Layer'].append(lyr)
            cell_regression['R^2'].append(r2)
            cell_regression['MSE'].append(mse)
            cell_regression['type'].append('Normal')


            if r2 > best_r2:
                best_r2 = r2
                best_r2_reg = reg
        if best_r2_reg is not None:
            print('CELL LAYER ',lyr)
            print('BEST REGRESSION R^2 ON HOLD OUT :: ',best_r2)

            mean_coef = best_r2_reg.coef_.mean()
            std_coef = best_r2_reg.coef_.std()

            print('STATISTICALLY SIGNIFICANT UNITS')
            print(np.where(np.abs(best_r2_reg.coef_) >= mean_coef + 3*std_coef )[0])
        else:
            print('UNABLE TO REGRESS')

    # CONTROL 2 DEPTH TARGETS
    for lyr in cell_states.keys():
        all_indices = np.arange(len(cell_states[lyr]))
        np.random.shuffle(all_indices)

        for cidx in range(args.cross_validation):
            reg,r2,mse = ridge_regression(cidx,args.cross_validation,cell_states[lyr],depth_targets,all_indices)

            cell_regression['Layer'].append(lyr)
            cell_regression['R^2'].append(r2)
            cell_regression['MSE'].append(mse)
            cell_regression['type'].append('Control 2 (Depth From Beginning of Sentence)')



    ax = sns.lineplot(x="Layer", y="R^2", hue='type', data=cell_regression)
    plt.title('Cell State R^2 Score Over Layers')
    plt.xlabel('Layer')
    plt.xticks(range(len(cell_states.keys())))
    plt.ylabel('R^2 Score')
    plt.legend(loc='upper right')
    plt.savefig('generated_files/imgs/cell_r2_layers_ctrl2.png')
    plt.close()

    ax = sns.lineplot(x="Layer", y="MSE", hue='type', data=cell_regression)
    plt.title('Cell State MSE Over Layers')
    plt.xlabel('Layer')
    plt.xticks(range(len(cell_states.keys())))
    plt.ylabel('MSE')
    plt.legend(loc='upper right')
    plt.savefig('generated_files/imgs/cell_mse_layers_ctrl2.png')
    plt.close()

    # CONTROL 1 SHUFFLE TARGETS
    random_shuffled_targets = targets.copy()
    np.random.shuffle(random_shuffled_targets)
    for lyr in cell_states.keys():
        all_indices = np.arange(len(cell_states[lyr]))
        np.random.shuffle(all_indices)

        for cidx in range(args.cross_validation):
            reg,r2,mse = ridge_regression(cidx,args.cross_validation,cell_states[lyr],random_shuffled_targets,all_indices)

            cell_regression['Layer'].append(lyr)
            cell_regression['R^2'].append(r2)
            cell_regression['MSE'].append(mse)
            cell_regression['type'].append('Control 1 (Randomly Shuffled Targets)')

    ax = sns.lineplot(x="Layer", y="R^2", hue='type', data=cell_regression)
    plt.title('Cell State R^2 Score Over Layers')
    plt.xlabel('Layer')
    plt.xticks(range(len(cell_states.keys())))
    plt.ylabel('R^2 Score')
    plt.legend(loc='upper right')
    plt.savefig('generated_files/imgs/cell_r2_layers.png')
    plt.close()

    ax = sns.lineplot(x="Layer", y="MSE", hue='type', data=cell_regression)
    plt.title('Cell State MSE Over Layers')
    plt.xlabel('Layer')
    plt.xticks(range(len(cell_states.keys())))
    plt.ylabel('MSE')
    plt.legend(loc='upper right')
    plt.savefig('generated_files/imgs/cell_mse_layers.png')
    plt.close()


# PCA ANALYSIS
if pca_analysis:
    for key in merged_hidden_states_eos:
        lyr_hiddens = merged_hidden_states_eos[key]

        pca = skd.PCA()
        pca.fit(lyr_hiddens)

        plt.plot(range(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_,label='Merged Hidden States <eos>')

        lyr_hiddens = merged_hidden_states_mid[key]
        pca = skd.PCA()
        pca.fit(lyr_hiddens)

        plt.plot(range(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_,label='Merged Hidden States <mid>')

        lyr_hiddens = other_hidden_states[key]
        pca = skd.PCA()
        pca.fit(lyr_hiddens)

        plt.plot(range(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_,label='Other Hidden States')

        plt.xlabel('PC Dimension')
        plt.ylabel('Normalized Eigenvalue')
        plt.title('Normalized Eigenvalue Significance of Hidden States')
        plt.legend()
        plt.savefig('generated_files/imgs/pc_variance_ratio_'+str(key)+'.png')
        plt.close()



if args.gated_forward:
# Differece GAS
    for lyr in range(len(forget_gates)):
        print('\n\n\nLAYER ',lyr+1)

        # Forget Gates

        f_layer_gates = forget_gates[lyr]
        i_layer_gates = input_gates[lyr]
        o_layer_gates = output_gates[lyr]

        hiddens = hidden_gates[lyr]
        cells = cell_gates[lyr]

        merge_values_pre = []
        other_values_pre = []
        merge_values_post_mid = []
        merge_values_post_eos = []
        other_values_post = []

        merge_values_abs_mid = []
        merge_values_abs_eos = []
        other_values_abs = []

        merge_values_nrm_mid = []
        merge_values_nrm_eos = []
        other_values_nrm = []

        merge_gates_eos = []
        merge_gates_mid = []
        other_gates = []

        delta_merge_gates_eos = []
        delta_merge_gates_mid = []
        delta_other_gates = []

        activity_data_dahaene = {'type':[],'activity':[],'label':[], 'depth':[]}

        activity_data_transition = {'type':[],'activity':[],'label':[]}

        for sidx in range(len(sentences)):
            sentence = sentences[sidx]
            pre_label = labels[sidx][:len(labels[sidx])-1]
            post_label = labels[sidx][1:]

            gates = f_layer_gates[sidx].mean(1)
            diffs = gates[1:] - gates[:len(gates)-1]

            abs = np.abs(gates)

            depth_change_locs = []

            # Are merge values statistically significant than the other values?
            pre_l = -1
            for lidx,l in enumerate(pre_label[:len(pre_label)-1]):
                if l <= pre_l:
                    # MERGE
                    merge_values_pre.append(diffs[lidx])
                    depth_change_locs.append(lidx)

                else:
                    other_values_pre.append(diffs[lidx])
                pre_l = l
            #
            # for x_line in depth_change_locs:
            #     plt.axvline(x=x_line,color='k')
            #
            # plt.title('Pre Labelling')
            # plt.plot(diffs)
            # plt.xticks(range(len(diffs)),sentence.split(' ')[:len(labels[sidx])-1])
            # plt.show()


            pre_l = -1
            for lidx,l in enumerate(post_label[:len(post_label)-1]):
                if l <= pre_l:
                    # MERGE
                    merge_values_post_mid.append(diffs[lidx])
                else:
                    other_values_post.append(diffs[lidx])
                pre_l = l
            merge_values_post_eos.append(diffs[len(diffs)-1])


            # abs
            pre_l = -1
            for lidx,l in enumerate(labels[sidx]):
                if l <= pre_l:
                    # MERGE
                    merge_values_abs_mid.append(abs[lidx])

                    merge_gates_mid.append(f_layer_gates[sidx][lidx])

                    if lidx > 0:
                        delta_merge_gates_mid.append(f_layer_gates[sidx][lidx] - f_layer_gates[sidx][lidx-1])
                else:
                    other_values_abs.append(abs[lidx])
                    other_gates.append(f_layer_gates[sidx][lidx])
                    if lidx > 0:
                        delta_other_gates.append(f_layer_gates[sidx][lidx] - f_layer_gates[sidx][lidx-1])
                pre_l = l
            merge_values_abs_eos.append(abs[len(abs)-1])
            merge_gates_eos.append(f_layer_gates[sidx][len(abs)-1])
            delta_merge_gates_eos.append(f_layer_gates[sidx][lidx] - f_layer_gates[sidx][lidx-1])


            # nrm
            pre_l = -1
            for lidx,l in enumerate(labels[sidx]):
                if l <= pre_l:
                    # MERGE
                    merge_values_nrm_mid.append(gates[lidx])
                else:
                    other_values_nrm.append(gates[lidx])
                pre_l = l
            merge_values_nrm_eos.append(gates[len(gates)-1])

            pre_label = '00'
            for lidx,l in enumerate(labels[sidx]):
                activity_data_dahaene['type'].append('Forget Gate')
                activity_data_dahaene['activity'].append(gates[lidx])
                activity_data_dahaene['label'].append(l)
                activity_data_dahaene['depth'].append(lidx+1)


                pre_label = (pre_label+hex(l)[2:])[1:]
                activity_data_transition['type'].append('Forget Gate')
                activity_data_transition['activity'].append(gates[lidx])
                activity_data_transition['label'].append(pre_label)

            pre_label = '00'
            gates = o_layer_gates[sidx].mean(1)
            for lidx,l in enumerate(labels[sidx]):
                activity_data_dahaene['type'].append('Output Gate')
                activity_data_dahaene['activity'].append(gates[lidx])
                activity_data_dahaene['label'].append(l)
                activity_data_dahaene['depth'].append(lidx+1)

                pre_label = (pre_label+hex(l)[2:])[1:]
                activity_data_transition['type'].append('Output Gate')
                activity_data_transition['activity'].append(gates[lidx])
                activity_data_transition['label'].append(pre_label)

            pre_label = '00'
            gates = i_layer_gates[sidx].mean(1)
            for lidx,l in enumerate(labels[sidx]):
                activity_data_dahaene['type'].append('Input Gate')
                activity_data_dahaene['activity'].append(gates[lidx])
                activity_data_dahaene['label'].append(l)
                activity_data_dahaene['depth'].append(lidx+1)

                pre_label = (pre_label+hex(l)[2:])[1:]
                activity_data_transition['type'].append('Input Gate')
                activity_data_transition['activity'].append(gates[lidx])
                activity_data_transition['label'].append(pre_label)

            # hidden states
            pre_label = '00'
            hidden = hiddens[sidx].mean(1)
            for lidx,l in enumerate(labels[sidx]):
                activity_data_dahaene['type'].append('Hidden State')
                activity_data_dahaene['activity'].append(hidden[lidx])
                activity_data_dahaene['label'].append(l)
                activity_data_dahaene['depth'].append(lidx+1)


                pre_label = (pre_label+hex(l)[2:])[1:]
                activity_data_transition['type'].append('Hidden State')
                activity_data_transition['activity'].append(hidden[lidx])
                activity_data_transition['label'].append(pre_label)

            # cell states
            pre_label = '00'
            cell = cells[sidx].mean(1)
            for lidx,l in enumerate(labels[sidx]):
                activity_data_dahaene['type'].append('Cell State')
                activity_data_dahaene['activity'].append(cell[lidx])
                activity_data_dahaene['label'].append(l)
                activity_data_dahaene['depth'].append(lidx+1)


                pre_label = (pre_label+hex(l)[2:])[1:]
                activity_data_transition['type'].append('Cell Gate')
                activity_data_transition['activity'].append(cell[lidx])
                activity_data_transition['label'].append(pre_label)

        ax = sns.lineplot(x="label", y="activity", hue='type', data=activity_data_dahaene)
        plt.title('Dahaene Depth Activation (Layer '+str(lyr+1)+')')
        plt.xlabel('Dahaene Depth')
        plt.ylabel('Average Activation')
        plt.savefig('generated_files/imgs/dahaene_activation_over_layer_'+str(lyr+1)+'.png')
        plt.close()

        ax = sns.lineplot(x="depth", y="activity", hue='type', data=activity_data_dahaene)
        plt.title('Sentence Depth Activation (Layer '+str(lyr+1)+')')
        plt.xlabel('Sentence Depth')
        plt.ylabel('Average Activation')
        plt.savefig('generated_files/imgs/depth_activation_over_layer_'+str(lyr+1)+'.png')
        plt.close()

        # assuming we are passed in a sorted list of activities
        sorted_labels = sorted(list(set(activity_data_transition['label'])))
        delta = 1
        activity_data_transition_df = pd.DataFrame.from_dict(activity_data_transition)
        delta_average_transition = {'delta':[],'activity':[],'type':[]}

        delta_labels = []
        for idx in range(len(activity_data_transition['label'])):
            unit = activity_data_transition['label'][idx]
            activity = activity_data_transition['activity'][idx]
            type = activity_data_transition['type'][idx]

            base = int(unit[0],16)
            cont = int(unit[1],16)

            delta_average_transition['delta'].append(cont - base)
            delta_average_transition['activity'].append(activity)
            delta_average_transition['type'].append(type)



        ax = sns.pointplot(x="delta", y="activity", data=delta_average_transition,hue='type', join=False)
        plt.title('Average Activation Merge Difference (Layer '+str(lyr+1)+')')
        plt.xlabel('Delta')
        plt.ylabel('Average Activation')
        # ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
        ax.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True)
        plt.tight_layout()

        # ax.set_xticklabels(delta_average_transition['delta'])
        # plt.show()
        plt.close()

        del_df = pd.DataFrame.from_dict(delta_average_transition)

        ax = sns.pairplot(x_vars="delta", y_vars="activity", data=del_df,hue='type')
        plt.title('Average Activation Merge Difference (Layer '+str(lyr+1)+')')
        plt.xlabel('Delta')
        plt.ylabel('Average Activation')


        # ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
        # ax.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True)
        plt.tight_layout()
        # plt.show()
        plt.close()


        shift = np.ceil(len(del_df.groupby('type'))/2)
        total_split = len(del_df.groupby('type'))+2
        magnifier = 1
        sub_idx = 0
        shifts = set()
        all_ds = set()

        subgroups = del_df.groupby('type')

        min_del = del_df['delta'].min()
        max_del = del_df['delta'].max()

        min_activity = del_df['activity'].min()
        max_activity = del_df['activity'].max()

        subdf_fig, subdf_axes = plt.subplots(len(subgroups),max_del-min_del+1)

        colors = ['#e67e22','#e74c3c','#9b59b6','#3498db','#2ecc71']

        for name,subdf in subgroups:
            x_vals = []
            y_vals = []

            idx_shifted = (1/total_split)*(1+sub_idx-shift)
            jitter_limit = 1/(2*total_split)
            shifts.add((1+sub_idx-shift))

            activities = {}

            for i in range(len(subdf)):

                sub_row = subdf.iloc[i]

                a = sub_row['activity']
                d = sub_row['delta']

                # d = d+idx_shifted
                all_ds.add(d)
                # jitter = np.random.uniform(low=-jitter_limit,high=jitter_limit,size=1)
                jitter = idx_shifted
                if d in activities:
                    activities[d].append(float(a))
                else:
                    activities[d] = []
                    activities[d].append(float(a))
                # x_vals.append((int(d)+jitter)*magnifier)
                # y_vals.append(float(a))
            for d in activities:
                subdf_axes[sub_idx,d-min_del].hist(activities[d],bins=25,color=colors[sub_idx],orientation='horizontal')
                subdf_axes[sub_idx,d-min_del].set_xticklabels([])
                subdf_axes[sub_idx,d-min_del].set_yticklabels([])

                subdf_axes[sub_idx,d-min_del].set_ybound(min_activity,max_activity)
                # axes[sub_idx,d-min_del].set_ybound(y_lower,y_upper)

                subdf_axes[sub_idx,0].set_ylabel(name)
                subdf_axes[len(colors)-1,d-min_del].set_xlabel('del '+str(d))

            sub_idx+=1

        # plt.legend(loc=(1.04,0))
        # plt.tight_layout()
        plt.suptitle('Gate Activation Distribution (Layer '+str(lyr+1)+')')
        plt.show()
        plt.close()
        import pdb; pdb.set_trace()



        merge_values_pre = np.array(merge_values_pre)
        other_values_pre = np.array(other_values_pre)

        merge_values_post_eos = np.array(merge_values_post_eos)
        merge_values_post_mid = np.array(merge_values_post_mid)
        other_values_post = np.array(other_values_post)

        merge_values_abs_eos = np.array(merge_values_abs_eos)
        merge_values_abs_mid = np.array(merge_values_abs_mid)
        other_values_abs = np.array(other_values_abs)

        merge_values_nrm_eos = np.array(merge_values_nrm_eos)
        merge_values_nrm_mid = np.array(merge_values_nrm_mid)
        other_values_nrm = np.array(other_values_nrm)

        max_value = max(np.max(merge_values_nrm_eos), np.max(merge_values_nrm_mid), np.max(other_values_nrm))

        merge_values_nrm_eos = merge_values_nrm_eos/max_value
        merge_values_nrm_mid = merge_values_nrm_mid/max_value
        other_values_nrm = other_values_nrm/max_value

        merge_gates_eos = np.array(merge_gates_eos)
        merge_gates_mid = np.array(merge_gates_mid)
        other_gates = np.array(other_gates)

        delta_merge_gates_eos = np.array(delta_merge_gates_eos)
        delta_merge_gates_mid = np.array(delta_merge_gates_mid)
        delta_other_gates = np.array(delta_other_gates)

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
        plt.savefig(args.save_folder+'/imgs/forget_plots_pre_delta_'+str(lyr+1)+'.png')
        plt.close()




        ks, p_val = stats.ks_2samp(other_values_post, merge_values_post_mid)
        print('\nPOST LABELLING')
        print('Kolmogorov-Smirnoff (MID) p-value::',p_val)
        ts, p_val = stats.ttest_1samp(merge_values_post_mid,other_values_post.mean())
        print('T-Test (MID) p-value::',p_val)

        ks, p_val = stats.ks_2samp(other_values_post, merge_values_post_eos)
        print('\nPOST LABELLING')
        print('Kolmogorov-Smirnoff (EOS) p-value::',p_val)
        ts, p_val = stats.ttest_1samp(merge_values_post_eos,other_values_post.mean())
        print('T-Test (EOS) p-value::',p_val)

        plt.title('Layer '+str(lyr+1)+':\n Comparing Distribution of Merge (Post Labelling) vs Other Forget Statistic\n(p-value = '+str(round(p_val,5))+')')
        plt.ylabel('Frequency')
        plt.xlabel('Statistic Value')
        plt.hist(other_values_post,bins=25,label='Other Values')
        plt.hist(merge_values_post_eos,bins=25,label='Merge Values <eos>')
        plt.hist(merge_values_post_mid,bins=25,label='Merge Values <mid>')
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.save_folder+'/imgs/forget_plots_post_delta_'+str(lyr+1)+'.png')
        plt.close()


        # ABS SIGNIFICANCE
        print('\nABS OF GATE')
        ks, p_val = stats.ks_2samp(other_values_abs, merge_values_abs_mid)
        print('\nABS LABELLING')
        print('Kolmogorov-Smirnoff (MID) p-value::',p_val)
        ts, p_val = stats.ttest_1samp(merge_values_abs_mid,other_values_abs.mean())
        print('T-Test (MID) p-value::',p_val)

        ks, p_val = stats.ks_2samp(other_values_abs, merge_values_abs_eos)
        print('\nABS LABELLING')
        print('Kolmogorov-Smirnoff (EOS) p-value::',p_val)
        ts, p_val = stats.ttest_1samp(merge_values_abs_eos,other_values_abs.mean())
        print('T-Test (EOS) p-value::',p_val)

        plt.title('Layer '+str(lyr+1)+':\n Comparing Distribution of Merge (ABS Labelling) vs Other Forget Statistic\n(p-value = '+str(round(p_val,5))+')')
        plt.ylabel('Frequency')
        plt.xlabel('Statistic Value')
        plt.hist(other_values_abs,bins=25,label='Other Values')
        plt.hist(merge_values_abs_eos,bins=25,label='Merge Values <eos>')
        plt.hist(merge_values_abs_mid,bins=25,label='Merge Values <mid>')
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.save_folder+'/imgs/forget_plots_abs_'+str(lyr+1)+'.png')
        plt.close()



        # NRM SIGNIFICANCE
        print('\nNRM OF GATE')
        ks, p_val = stats.ks_2samp(other_values_nrm, merge_values_nrm_mid)
        print('\nNRM LABELLING')
        print('Kolmogorov-Smirnoff (MID) p-value::',p_val)
        ts, p_val = stats.ttest_1samp(merge_values_nrm_mid,other_values_nrm.mean())
        print('T-Test (MID) p-value::',p_val)

        ks, p_val = stats.ks_2samp(other_values_nrm, merge_values_nrm_eos)
        print('\nNRM LABELLING')
        print('Kolmogorov-Smirnoff (EOS) p-value::',p_val)
        ts, p_val = stats.ttest_1samp(merge_values_nrm_eos,other_values_nrm.mean())
        print('T-Test (EOS) p-value::',p_val)

        plt.title('Layer '+str(lyr+1)+':\n Comparing Distribution of Merge (NRM Labelling) vs Other Forget Statistic\n(p-value = '+str(round(p_val,5))+')')
        plt.ylabel('Frequency')
        plt.xlabel('Statistic Value')
        plt.hist(other_values_nrm,bins=25,label='Other Values')
        plt.hist(merge_values_nrm_eos,bins=25,label='Merge Values <eos>')
        plt.hist(merge_values_nrm_mid,bins=25,label='Merge Values <mid>')
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.save_folder+'/imgs/forget_plots_nrm_'+str(lyr+1)+'.png')
        plt.close()



        pca = skd.PCA()
        pca.fit(merge_gates_eos)
        plt.plot(range(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_,label='Merged Forget Gates <eos>')

        eos_cdf = [0]
        for pidx,v in enumerate(pca.explained_variance_ratio_):
            eos_cdf.append(eos_cdf[pidx] + v)

        pca = skd.PCA()
        pca.fit(merge_gates_mid)
        plt.plot(range(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_,label='Merged Forget Gates <mid>')

        mid_cdf = [0]
        for pidx,v in enumerate(pca.explained_variance_ratio_):
            mid_cdf.append(mid_cdf[pidx] + v)

        pca = skd.PCA()
        pca.fit(other_gates)
        plt.plot(range(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_,label='Other Forget Gates')

        oth_cdf = [0]
        for pidx,v in enumerate(pca.explained_variance_ratio_):
            oth_cdf.append(oth_cdf[pidx] + v)

        plt.xlabel('PC Dimension')
        plt.ylabel('Normalized Eigenvalue')
        plt.title('Normalized Eigenvalue Significance of Forget Gates')
        plt.legend()
        plt.savefig('generated_files/imgs/pc_variance_ratio_forgetG_'+str(lyr)+'.png')
        plt.close()


        plt.plot(range(len(eos_cdf)),eos_cdf,label='Merge Forget Gates <eos>')
        plt.plot(range(len(mid_cdf)),mid_cdf,label='Merge Forget Gates <mid>')
        plt.plot(range(len(oth_cdf)),oth_cdf,label='Other Forget Gates')
        plt.legend()
        plt.title('Normalized Eigenvalue Significance of Forget Gates CDF')
        plt.savefig('generated_files/imgs/pc_cdf_forgetG_'+str(lyr)+'.png')
        plt.close()



        # DIFF PCA

        pca = skd.PCA()
        pca.fit(delta_merge_gates_eos)
        plt.plot(range(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_,label='Merged Delta Forget Gates <eos>')

        eos_cdf = [0]
        for pidx,v in enumerate(pca.explained_variance_ratio_):
            eos_cdf.append(eos_cdf[pidx] + v)

        pca = skd.PCA()
        pca.fit(delta_merge_gates_mid)
        plt.plot(range(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_,label='Merged Delta Forget Gates <mid>')

        mid_cdf = [0]
        for pidx,v in enumerate(pca.explained_variance_ratio_):
            mid_cdf.append(mid_cdf[pidx] + v)

        pca = skd.PCA()
        pca.fit(delta_other_gates)
        plt.plot(range(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_,label='Other Delta Forget Gates')

        oth_cdf = [0]
        for pidx,v in enumerate(pca.explained_variance_ratio_):
            oth_cdf.append(oth_cdf[pidx] + v)

        plt.xlabel('PC Dimension')
        plt.ylabel('Normalized Eigenvalue')
        plt.title('Normalized Eigenvalue Significance of Delta Forget Gates Layer '+str(lyr+1))
        plt.legend()
        plt.savefig('generated_files/imgs/pc_variance_ratio_delta_forgetG_'+str(lyr)+'.png')
        plt.close()


        plt.plot(range(len(eos_cdf)),eos_cdf,label='Merge Forget Gates <eos>')
        plt.plot(range(len(mid_cdf)),mid_cdf,label='Merge Forget Gates <mid>')
        plt.plot(range(len(oth_cdf)),oth_cdf,label='Other Forget Gates')
        plt.legend()
        plt.title('Normalized Eigenvalue Significance of Delta Forget Gates CDF Layer '+str(lyr+1))
        plt.savefig('generated_files/imgs/pc_cdf_delta_forgetG_'+str(lyr)+'.png')
        plt.close()


        # all = np.vstack((merge_gates_eos,merge_gates_mid))
        # all = np.vstack((all, other_gates))
        #
        # import pdb; pdb.set_trace()
        #
        # pca = PCA()
        # pca.fit(all)








# END OF FILE
