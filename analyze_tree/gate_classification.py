import torch
import torch.nn as nn

from load_model import load_model
from load_data import load_data

import sklearn as sk
import sklearn.linear_model as skl
import sklearn.metrics as skm
import sklearn.decomposition as skd

import numpy as np

import data
from tqdm import tqdm

import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse

import utils

import os

import seaborn as sns

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description='Suprisal plot generator')
parser.add_argument('--observe', type=str, default='forget_gate',
                    help='Area in which we will be looking for important statistics {forget_gate|input_gate|output_gate|cell_state|hidden_state}')
parser.add_argument('--classify', default='False', action='store_true',
                    help='Determines whether or not we will run classification on the observed gate and predict whether or not it is a   merge vs. not merge  gate.')
parser.add_argument('--controls', default='False', action='store_true',
                    help='Only matters if --classification is set to True. If this is True then we will train classifiers on control scenarios to compare to the accuracies.')
parser.add_argument('--train_percent', default='0.8', type=float,
                    help='Only matters if --classification is set to True. Determines what percent of the words will be used to train.')
args = parser.parse_args()

train_percent = 0.8
# import pdb; pdb.set_trace()
sentences,labels = load_data('dahaene_dataset/filtered_sentences.txt','open_nodes')

if args.observe == 'forget_gate':
    super_title = "Forget Gate"
    all_gates = np.load('generated_files/forget_gates.npy').item()
elif args.observe == 'output_gate':
    super_title = "Output Gate"
    all_gates = np.load('generated_files/output_gates.npy').item()
elif args.observe == 'input_gate':
    super_title = "Input Gate"
    all_gates = np.load('generated_files/input_gates.npy').item()
elif args.observe == 'cell_state':
    super_title = "Cell State"
    all_gates = np.load('generated_files/cell_gates.npy').item()
elif args.observe == 'hidden_state':
    super_title = "Hidden State"
    all_gates = np.load('generated_files/hidden_gates.npy').item()

C = [(255/255,0/255,176/255),(253/255,0/255,175/255),(251/255,0/255,173/255),(249/255,0/255,172/255),(247/255,0/255,171/255),(245/255,0/255,169/255),(243/255,0/255,168/255),(241/255,0/255,167/255),(239/255,0/255,165/255),(236/255,0/255,164/255),(234/255,0/255,162/255),(232/255,0/255,161/255),(230/255,0/255,160/255),(228/255,0/255,158/255),(226/255,0/255,157/255),(224/255,0/255,156/255),(222/255,0/255,154/255),(220/255,0/255,153/255),(218/255,0/255,152/255),(216/255,0/255,150/255),(214/255,0/255,149/255),(212/255,0/255,147/255),(210/255,0/255,146/255),(208/255,0/255,145/255),(206/255,0/255,143/255),(203/255,0/255,142/255),(201/255,0/255,141/255),(199/255,0/255,139/255),(197/255,0/255,138/255),(195/255,0/255,136/255),(193/255,0/255,135/255),(191/255,0/255,134/255),(189/255,0/255,132/255),(187/255,0/255,131/255),(185/255,0/255,130/255),(183/255,0/255,128/255),(181/255,0/255,127/255),(179/255,0/255,126/255),(177/255,0/255,124/255),(175/255,0/255,123/255),(173/255,0/255,121/255),(171/255,0/255,120/255),(168/255,0/255,119/255),(166/255,0/255,117/255),(164/255,0/255,116/255),(162/255,0/255,115/255),(160/255,0/255,113/255),(158/255,0/255,112/255),(156/255,0/255,110/255),(154/255,0/255,109/255),(152/255,0/255,108/255),(150/255,0/255,106/255),(148/255,0/255,105/255),(146/255,0/255,104/255),(144/255,0/255,102/255),(142/255,0/255,101/255),(140/255,0/255,99/255),(138/255,0/255,98/255),(135/255,0/255,97/255),(133/255,0/255,95/255),(131/255,0/255,94/255),(129/255,0/255,93/255),(127/255,0/255,91/255),(125/255,0/255,90/255),(123/255,0/255,89/255),(121/255,0/255,87/255),(119/255,0/255,86/255),(117/255,0/255,84/255),(115/255,0/255,83/255),(113/255,0/255,82/255),(112/255,0/255,80/255),(110/255,0/255,79/255),(108/255,0/255,77/255),(106/255,0/255,76/255),(104/255,0/255,75/255),(102/255,0/255,73/255),(100/255,0/255,72/255),(98/255,0/255,70/255),(96/255,0/255,69/255),(94/255,0/255,68/255),(92/255,0/255,66/255),(90/255,0/255,65/255),(88/255,0/255,63/255),(86/255,0/255,62/255),(84/255,0/255,61/255),(82/255,0/255,59/255),(80/255,0/255,58/255),(79/255,0/255,56/255),(77/255,0/255,55/255),(75/255,0/255,54/255),(73/255,0/255,52/255),(71/255,0/255,51/255),(69/255,0/255,49/255),(67/255,0/255,48/255),(65/255,0/255,47/255),(63/255,0/255,45/255),(61/255,0/255,44/255),(59/255,0/255,43/255),(57/255,0/255,41/255),(55/255,0/255,40/255),(53/255,0/255,38/255),(51/255,0/255,37/255),(49/255,0/255,36/255),(48/255,0/255,34/255),(46/255,0/255,33/255),(44/255,0/255,31/255),(42/255,0/255,30/255),(40/255,0/255,29/255),(38/255,0/255,27/255),(36/255,0/255,26/255),(34/255,0/255,24/255),(32/255,0/255,23/255),(30/255,0/255,22/255),(28/255,0/255,20/255),(26/255,0/255,19/255),(24/255,0/255,17/255),(22/255,0/255,16/255),(20/255,0/255,15/255),(18/255,0/255,13/255),(16/255,0/255,12/255),(15/255,0/255,10/255),(13/255,0/255,9/255),(11/255,0/255,8/255),(9/255,0/255,6/255),(7/255,0/255,5/255),(5/255,0/255,3/255),(3/255,0/255,2/255),(1/255,0/255,1/255),(0/255,1/255,1/255),(0/255,2/255,3/255),(0/255,4/255,5/255),(0/255,6/255,7/255),(0/255,7/255,9/255),(0/255,9/255,11/255),(0/255,10/255,13/255),(0/255,12/255,15/255),(0/255,13/255,16/255),(0/255,15/255,18/255),(0/255,17/255,20/255),(0/255,18/255,22/255),(0/255,20/255,24/255),(0/255,21/255,26/255),(0/255,23/255,28/255),(0/255,25/255,30/255),(0/255,26/255,32/255),(0/255,28/255,34/255),(0/255,29/255,36/255),(0/255,31/255,38/255),(0/255,33/255,40/255),(0/255,34/255,42/255),(0/255,36/255,44/255),(0/255,37/255,46/255),(0/255,39/255,48/255),(0/255,40/255,49/255),(0/255,42/255,51/255),(0/255,44/255,53/255),(0/255,45/255,55/255),(0/255,47/255,57/255),(0/255,48/255,59/255),(0/255,50/255,61/255),(0/255,52/255,63/255),(0/255,53/255,65/255),(0/255,55/255,67/255),(0/255,56/255,69/255),(0/255,58/255,71/255),(0/255,60/255,73/255),(0/255,61/255,75/255),(0/255,63/255,77/255),(0/255,64/255,79/255),(0/255,66/255,80/255),(0/255,67/255,82/255),(0/255,69/255,84/255),(0/255,71/255,86/255),(0/255,72/255,88/255),(0/255,74/255,90/255),(0/255,75/255,92/255),(0/255,77/255,94/255),(0/255,79/255,96/255),(0/255,80/255,98/255),(0/255,82/255,100/255),(0/255,83/255,102/255),(0/255,85/255,104/255),(0/255,87/255,106/255),(0/255,88/255,108/255),(0/255,90/255,110/255),(0/255,91/255,112/255),(0/255,93/255,113/255),(0/255,94/255,115/255),(0/255,96/255,117/255),(0/255,98/255,119/255),(0/255,99/255,121/255),(0/255,101/255,123/255),(0/255,102/255,125/255),(0/255,104/255,127/255),(0/255,105/255,129/255),(0/255,106/255,131/255),(0/255,107/255,133/255),(0/255,109/255,135/255),(0/255,110/255,137/255),(0/255,111/255,140/255),(0/255,113/255,142/255),(0/255,114/255,144/255),(0/255,115/255,146/255),(0/255,117/255,148/255),(0/255,118/255,150/255),(0/255,119/255,152/255),(0/255,121/255,154/255),(0/255,122/255,156/255),(0/255,123/255,158/255),(0/255,124/255,160/255),(0/255,126/255,162/255),(0/255,127/255,164/255),(0/255,128/255,166/255),(0/255,130/255,168/255),(0/255,131/255,170/255),(0/255,132/255,172/255),(0/255,134/255,174/255),(0/255,135/255,176/255),(0/255,136/255,178/255),(0/255,138/255,180/255),(0/255,139/255,183/255),(0/255,140/255,185/255),(0/255,141/255,187/255),(0/255,143/255,189/255),(0/255,144/255,191/255),(0/255,145/255,193/255),(0/255,147/255,195/255),(0/255,148/255,197/255),(0/255,149/255,199/255),(0/255,151/255,201/255),(0/255,152/255,203/255),(0/255,153/255,205/255),(0/255,155/255,207/255),(0/255,156/255,209/255),(0/255,157/255,211/255),(0/255,158/255,213/255),(0/255,160/255,215/255),(0/255,161/255,217/255),(0/255,162/255,219/255),(0/255,164/255,221/255),(0/255,165/255,224/255),(0/255,166/255,226/255),(0/255,168/255,228/255),(0/255,169/255,230/255),(0/255,170/255,232/255),(0/255,172/255,234/255),(0/255,173/255,236/255),(0/255,174/255,238/255),(0/255,176/255,240/255),(0/255,177/255,242/255),(0/255,178/255,244/255),(0/255,179/255,246/255),(0/255,181/255,248/255),(0/255,182/255,250/255),(0/255,183/255,252/255),(0/255,185/255,254/255)]

C = [(255/255,0/255,0/255),(253/255,0/255,0/255),(251/255,0/255,0/255),(249/255,0/255,0/255),(247/255,0/255,0/255),(245/255,0/255,0/255),(243/255,0/255,0/255),(241/255,0/255,0/255),(239/255,0/255,0/255),(236/255,0/255,0/255),(234/255,0/255,0/255),(232/255,0/255,0/255),(230/255,0/255,0/255),(228/255,0/255,0/255),(226/255,0/255,0/255),(224/255,0/255,0/255),(222/255,0/255,0/255),(220/255,0/255,0/255),(218/255,0/255,0/255),(216/255,0/255,0/255),(214/255,0/255,0/255),(212/255,0/255,0/255),(210/255,0/255,0/255),(208/255,0/255,0/255),(206/255,0/255,0/255),(203/255,0/255,0/255),(201/255,0/255,0/255),(199/255,0/255,0/255),(197/255,0/255,0/255),(195/255,0/255,0/255),(193/255,0/255,0/255),(191/255,0/255,0/255),(189/255,0/255,0/255),(187/255,0/255,0/255),(185/255,0/255,0/255),(183/255,0/255,0/255),(181/255,0/255,0/255),(179/255,0/255,0/255),(177/255,0/255,0/255),(175/255,0/255,0/255),(173/255,0/255,0/255),(171/255,0/255,0/255),(168/255,0/255,0/255),(166/255,0/255,0/255),(164/255,0/255,0/255),(162/255,0/255,0/255),(160/255,0/255,0/255),(158/255,0/255,0/255),(156/255,0/255,0/255),(154/255,0/255,0/255),(152/255,0/255,0/255),(150/255,0/255,0/255),(148/255,0/255,0/255),(146/255,0/255,0/255),(144/255,0/255,0/255),(142/255,0/255,0/255),(140/255,0/255,0/255),(138/255,0/255,0/255),(135/255,0/255,0/255),(133/255,0/255,0/255),(131/255,0/255,0/255),(129/255,0/255,0/255),(127/255,0/255,0/255),(125/255,0/255,0/255),(123/255,0/255,0/255),(121/255,0/255,0/255),(119/255,0/255,0/255),(117/255,0/255,0/255),(115/255,0/255,0/255),(113/255,0/255,0/255),(112/255,0/255,0/255),(110/255,0/255,0/255),(108/255,0/255,0/255),(106/255,0/255,0/255),(104/255,0/255,0/255),(102/255,0/255,0/255),(100/255,0/255,0/255),(98/255,0/255,0/255),(96/255,0/255,0/255),(94/255,0/255,0/255),(92/255,0/255,0/255),(90/255,0/255,0/255),(88/255,0/255,0/255),(86/255,0/255,0/255),(84/255,0/255,0/255),(82/255,0/255,0/255),(80/255,0/255,0/255),(79/255,0/255,0/255),(77/255,0/255,0/255),(75/255,0/255,0/255),(73/255,0/255,0/255),(71/255,0/255,0/255),(69/255,0/255,0/255),(67/255,0/255,0/255),(65/255,0/255,0/255),(63/255,0/255,0/255),(61/255,0/255,0/255),(59/255,0/255,0/255),(57/255,0/255,0/255),(55/255,0/255,0/255),(53/255,0/255,0/255),(51/255,0/255,0/255),(49/255,0/255,0/255),(48/255,0/255,0/255),(46/255,0/255,0/255),(44/255,0/255,0/255),(42/255,0/255,0/255),(40/255,0/255,0/255),(38/255,0/255,0/255),(36/255,0/255,0/255),(34/255,0/255,0/255),(32/255,0/255,0/255),(30/255,0/255,0/255),(28/255,0/255,0/255),(26/255,0/255,0/255),(24/255,0/255,0/255),(22/255,0/255,0/255),(20/255,0/255,0/255),(18/255,0/255,0/255),(16/255,0/255,0/255),(15/255,0/255,0/255),(13/255,0/255,0/255),(11/255,0/255,0/255),(9/255,0/255,0/255),(7/255,0/255,0/255),(5/255,0/255,0/255),(3/255,0/255,0/255),(1/255,0/255,0/255),(0/255,0/255,1/255),(0/255,0/255,3/255),(0/255,0/255,5/255),(0/255,0/255,7/255),(0/255,0/255,9/255),(0/255,0/255,11/255),(0/255,0/255,13/255),(0/255,0/255,15/255),(0/255,0/255,16/255),(0/255,0/255,18/255),(0/255,0/255,20/255),(0/255,0/255,22/255),(0/255,0/255,24/255),(0/255,0/255,26/255),(0/255,0/255,28/255),(0/255,0/255,30/255),(0/255,0/255,32/255),(0/255,0/255,34/255),(0/255,0/255,36/255),(0/255,0/255,38/255),(0/255,0/255,40/255),(0/255,0/255,42/255),(0/255,0/255,44/255),(0/255,0/255,46/255),(0/255,0/255,48/255),(0/255,0/255,49/255),(0/255,0/255,51/255),(0/255,0/255,53/255),(0/255,0/255,55/255),(0/255,0/255,57/255),(0/255,0/255,59/255),(0/255,0/255,61/255),(0/255,0/255,63/255),(0/255,0/255,65/255),(0/255,0/255,67/255),(0/255,0/255,69/255),(0/255,0/255,71/255),(0/255,0/255,73/255),(0/255,0/255,75/255),(0/255,0/255,77/255),(0/255,0/255,79/255),(0/255,0/255,80/255),(0/255,0/255,82/255),(0/255,0/255,84/255),(0/255,0/255,86/255),(0/255,0/255,88/255),(0/255,0/255,90/255),(0/255,0/255,92/255),(0/255,0/255,94/255),(0/255,0/255,96/255),(0/255,0/255,98/255),(0/255,0/255,100/255),(0/255,0/255,102/255),(0/255,0/255,104/255),(0/255,0/255,106/255),(0/255,0/255,108/255),(0/255,0/255,110/255),(0/255,0/255,112/255),(0/255,0/255,113/255),(0/255,0/255,115/255),(0/255,0/255,117/255),(0/255,0/255,119/255),(0/255,0/255,121/255),(0/255,0/255,123/255),(0/255,0/255,125/255),(0/255,0/255,127/255),(0/255,0/255,129/255),(0/255,0/255,131/255),(0/255,0/255,133/255),(0/255,0/255,135/255),(0/255,0/255,137/255),(0/255,0/255,140/255),(0/255,0/255,142/255),(0/255,0/255,144/255),(0/255,0/255,146/255),(0/255,0/255,148/255),(0/255,0/255,150/255),(0/255,0/255,152/255),(0/255,0/255,154/255),(0/255,0/255,156/255),(0/255,0/255,158/255),(0/255,0/255,160/255),(0/255,0/255,162/255),(0/255,0/255,164/255),(0/255,0/255,166/255),(0/255,0/255,168/255),(0/255,0/255,170/255),(0/255,0/255,172/255),(0/255,0/255,174/255),(0/255,0/255,176/255),(0/255,0/255,178/255),(0/255,0/255,180/255),(0/255,0/255,183/255),(0/255,0/255,185/255),(0/255,0/255,187/255),(0/255,0/255,189/255),(0/255,0/255,191/255),(0/255,0/255,193/255),(0/255,0/255,195/255),(0/255,0/255,197/255),(0/255,0/255,199/255),(0/255,0/255,201/255),(0/255,0/255,203/255),(0/255,0/255,205/255),(0/255,0/255,207/255),(0/255,0/255,209/255),(0/255,0/255,211/255),(0/255,0/255,213/255),(0/255,0/255,215/255),(0/255,0/255,217/255),(0/255,0/255,219/255),(0/255,0/255,221/255),(0/255,0/255,224/255),(0/255,0/255,226/255),(0/255,0/255,228/255),(0/255,0/255,230/255),(0/255,0/255,232/255),(0/255,0/255,234/255),(0/255,0/255,236/255),(0/255,0/255,238/255),(0/255,0/255,240/255),(0/255,0/255,242/255),(0/255,0/255,244/255),(0/255,0/255,246/255),(0/255,0/255,248/255),(0/255,0/255,250/255),(0/255,0/255,252/255),(0/255,0/255,254/255)]

cm = mpl.colors.ListedColormap(C)

def closest_factors(num):
    closest_factors = []
    distance = num
    for i in range(2,int(num//2+1)):
        if num % i == 0:
            fc1 = i
            fc2 = num/i


            if abs(fc2 - fc1) <= distance:
                closest_factors = [int(fc1), int(fc2)]
                distance = abs(fc2 - fc1)

    return tuple(closest_factors)

def clip(x, limit):
    if limit is None:
        return x
    return np.clip(x,-limit,limit)

gate_activity_per_layer = {}
mean_per_layer = {}
std_per_layer = {}
for lyr in all_gates.keys():
    print(lyr)
    if lyr not in gate_activity_per_layer:
        gate_activity_per_layer[lyr] = {}

    merge_target = []

    gates_train_merge = []
    gates_train_not_merge = []
    gates_train = []

    sentences_with_context = []
    word_indices_with_cont = []

    pre_context_window = 5
    post_context_window = 1
    for sidx, sentence in enumerate(sentences):
        previous_label = 0
        last_merge = 0
        for lidx, label in enumerate(labels[sidx]):
            if label <= previous_label:
                # merge
                merge_target.append(1)
                gates_train_merge.append(all_gates[lyr][sidx][lidx])

                if lidx - last_merge >= pre_context_window and lidx < len(labels[sidx])-post_context_window:
                    sentences_with_context.append(sidx)
                    word_indices_with_cont.append(lidx)

                last_merge = lidx
            else:
                merge_target.append(0)
                gates_train_not_merge.append(all_gates[lyr][sidx][lidx])

            delta = label - previous_label
            if delta in gate_activity_per_layer[lyr]:
                gate_activity_per_layer[lyr][delta].append(all_gates[lyr][sidx][lidx])
            else:
                gate_activity_per_layer[lyr][delta] = []
                gate_activity_per_layer[lyr][delta].append(all_gates[lyr][sidx][lidx])


            previous_label = label
            gates_train.append(all_gates[lyr][sidx][lidx])
    difference_window = 1

    context_gates = []
    average_gates = []
    print('Contextual Sentences',len(sentences_with_context))
    for idx, sidx in enumerate(sentences_with_context):
        midx = word_indices_with_cont[idx]
        for lidx in range(midx-pre_context_window + 1 + difference_window, midx + 1 + post_context_window):
            difference = all_gates[lyr][sidx][lidx] - all_gates[lyr][sidx][lidx - difference_window]

            if len(context_gates) < lidx - midx + pre_context_window - difference_window:
                context_gates.append(difference)
            else:
                context_gates[lidx - midx + pre_context_window - 1 - difference_window] = (idx * (context_gates[lidx - midx + pre_context_window - 1 - difference_window]) + difference)/(idx + 1)

    for idx, sidx in enumerate(sentences_with_context):
        midx = word_indices_with_cont[idx]
        for lidx in range(midx-pre_context_window + 1, midx + 1 + post_context_window):
            if len(average_gates) < lidx - midx + pre_context_window:
                average_gates.append(all_gates[lyr][sidx][lidx])
            else:
                average_gates[lidx - midx + pre_context_window - 1] = (idx * (average_gates[lidx - midx + pre_context_window - 1]) + all_gates[lyr][sidx][lidx])/(idx + 1)

    # for end of sentence consider where you look. is it last word or one before? Also double check that that is what you are doing for the other files too

    gates_train = np.array(gates_train)

    gates_train_merge = np.array(gates_train_merge)
    gates_train_not_merge = np.array(gates_train_not_merge)

    ordered_gates = np.vstack((gates_train_not_merge,gates_train_merge))


    mean_gate = ordered_gates.mean(0)
    std_gate = ordered_gates.std(0)

    mean_per_layer[lyr] = mean_gate
    std_per_layer[lyr] = std_gate
    # This is a good way to show that the gates are statistically different
    z_score_gates = (ordered_gates - mean_gate)/std_gate


    vlength = len(context_gates[0])

    context_gates = np.array(context_gates)
    # import pdb; pdb.set_trace()
    mean_diff = context_gates.mean(0)
    std_diff = context_gates.std(0)
    # zscoring
    z_context_gates = (context_gates - mean_diff)/std_diff

    # 0 - 0.3136, 1 - 0.3615, 2 - 0.5622, 3 - 0.8087
    # sums
    # 0 - 0.7, 1 = 2 , 2 = -0.27, 3 = -8
    # Difference plots
    f, axarr = plt.subplots(nrows=3,ncols=pre_context_window-difference_window+post_context_window)

    z_clip_limit = None

    if z_clip_limit is None:
        max_z_score = np.max(np.abs(z_context_gates))
    else:
        max_z_score = z_clip_limit
    max_context = np.max(np.abs(context_gates))

    diff_avg = []
    diff_z_avg = []
    diff_abs = []

    for fidx in range(pre_context_window-difference_window + post_context_window):
        im0 = axarr[0,fidx].imshow(context_gates[fidx].reshape(closest_factors(vlength)),aspect='auto', vmin=-max_context, vmax=max_context)
        axarr[0,fidx].get_xaxis().set_ticks([])
        axarr[0,fidx].get_yaxis().set_ticks([])

        diff_avg.append(context_gates[fidx].mean())

        z_scored = clip(z_score_gates[fidx],z_clip_limit)

        im1 = axarr[1,fidx].imshow(z_scored.reshape(closest_factors(vlength)),aspect='auto',cmap='seismic', vmin=-max_z_score, vmax=max_z_score)

        axarr[1,fidx].get_xaxis().set_ticks([])
        axarr[1,fidx].get_yaxis().set_ticks([])

        diff_z_avg.append(z_scored.mean())



        im2 = axarr[2,fidx].imshow(np.abs(z_scored).reshape(closest_factors(vlength)),aspect='auto',cmap='seismic', vmin=-max_z_score, vmax=max_z_score)

        axarr[2,fidx].get_xaxis().set_ticks([])
        axarr[2,fidx].get_yaxis().set_ticks([])

        diff_abs.append(np.abs(z_scored).mean())


        title = 'Merge'
        if not( fidx == pre_context_window - difference_window - 1):
            title += ' ' + str(-(pre_context_window-difference_window - fidx-1))
        axarr[0,fidx].set_title(title)

        # figures[fidx].clim(0, 1)
        # figures[fidx].colorbar()
    # f.subplots_adjust(right=0.8)
    # cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    cbt = f.colorbar(im0, ax=axarr[0,:])
    cbt.set_clim(-max_context, max_context)

    cbb = f.colorbar(im1, ax=axarr[1,:])
    cbb.set_clim(-max_z_score, max_z_score)

    cbb = f.colorbar(im2, ax=axarr[2,:])
    cbb.set_clim(-max_z_score, max_z_score)

    axarr[0,0].set_ylabel('Difference')

    axarr[1,0].set_ylabel('Z-Scored Difference')

    axarr[2,0].set_ylabel('Absolute Z-Score')

    f.suptitle(super_title+' Difference Around Merge (Layer '+str(lyr+1)+')')

    plt.show()

    plt.plot(range(pre_context_window-difference_window + post_context_window),diff_z_avg,label='Z-Scored '+super_title+' Difference', marker="o", markersize=5)

    plt.plot(range(pre_context_window-difference_window + post_context_window),diff_avg,label=super_title+' Difference', marker="o", markersize=5)

    plt.plot(range(pre_context_window-difference_window + post_context_window),diff_abs,label='Absolute Z-Deviation', marker="o", markersize=5)

    plt.title('Average Activity Difference Accross Contextual Sentences in '+super_title+' (Layer '+str(lyr+1)+')')
    plt.xlabel('Distance From Merge')
    plt.ylabel('Average Activity Difference')
    plt.xticks(range(pre_context_window-difference_window + post_context_window),range(-pre_context_window+1+difference_window,post_context_window+1+difference_window))
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.close()



    # zscore gates
    average_gates = np.array(average_gates)

    z_average_gates = (average_gates - mean_gate)/std_gate

    max_avg_gate = np.max(np.abs(average_gates))
    max_zavg_gate = np.max(np.abs(z_average_gates))

    f, axarr = plt.subplots(nrows=3,ncols=pre_context_window+post_context_window)

    average_avg_gate = []
    average_avg_zscore = []
    average_abs_activity = []

    for fidx in range(pre_context_window+post_context_window):
        im0 = axarr[0,fidx].imshow(average_gates[fidx].reshape(closest_factors(vlength)),aspect='auto',cmap='seismic', vmin=0, vmax=1)

        title = 'Merge'
        if not( fidx == pre_context_window - 1):
            title += ' ' + str(-(pre_context_window - fidx-1))
        axarr[0,fidx].set_title(title)

        axarr[0,fidx].get_xaxis().set_ticks([])
        axarr[0,fidx].get_yaxis().set_ticks([])

        average_avg_gate.append(average_gates[fidx].mean())



        im1 = axarr[1,fidx].imshow(z_average_gates[fidx].reshape(closest_factors(vlength)),aspect='auto',cmap='seismic', vmin=-max_zavg_gate, vmax=max_zavg_gate)

        axarr[1,fidx].get_xaxis().set_ticks([])
        axarr[1,fidx].get_yaxis().set_ticks([])

        average_avg_zscore.append(z_average_gates[fidx].mean())



        im2 = axarr[2,fidx].imshow(np.abs(z_average_gates[fidx]).reshape(closest_factors(vlength)),aspect='auto',cmap='seismic', vmin=-max_zavg_gate, vmax=max_zavg_gate)

        axarr[2,fidx].get_xaxis().set_ticks([])
        axarr[2,fidx].get_yaxis().set_ticks([])

        average_abs_activity.append(np.abs(z_average_gates[fidx]).mean())

        # figures[fidx].clim(0, 1)
        # figures[fidx].colorbar()
    cbt = f.colorbar(im0, ax=axarr[0,:])
    cbt.set_clim(0, 1)

    cbb = f.colorbar(im1, ax=axarr[1,:])
    cbb.set_clim(-max_zavg_gate, max_zavg_gate)

    cbb = f.colorbar(im1, ax=axarr[2,:])
    cbb.set_clim(-max_zavg_gate, max_zavg_gate)
    axarr[0,0].set_ylabel('Averaged Regional Gate')

    axarr[1,0].set_ylabel('Z-Scored Gate')

    axarr[2,0].set_ylabel('Absolute Z-Score')

    f.suptitle(super_title+' Around Merge (Layer '+str(lyr+1)+')')

    plt.show()
    plt.close()

    plt.plot(range(pre_context_window+post_context_window),average_avg_zscore,label='Z-Scored '+super_title, marker="o", markersize=5)

    plt.plot(range(pre_context_window+post_context_window),average_avg_gate,label=super_title, marker="o", markersize=5)

    plt.plot(range(pre_context_window+post_context_window),average_abs_activity,label='Absolute Z-Deviation', marker="o", markersize=5)

    plt.title('Average Activity Accross Contextual Sentences in '+super_title+' (Layer '+str(lyr+1)+')')
    plt.xlabel('Distance From Merge')
    plt.ylabel('Average Activity')
    plt.xticks(range(pre_context_window+post_context_window),range(-pre_context_window+1,post_context_window+1))
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.close()
    import pdb; pdb.set_trace()
    if args.classify and False:
        train_indices = np.random.choice(range(len(gates_train_merge)),int(args.train_percent*len(gates_train_merge)),replace=False)
        X_train = gates_train_merge[train_indices,:]

        y_train = [1]*len(train_indices)

        test_indices = np.array(list(set(range(len(gates_train_merge))).difference(train_indices)))

        gates_merge_test = gates_train_merge[test_indices,:]

        train_indices = np.random.choice(range(len(gates_train_not_merge)),int(args.train_percent*len(gates_train_not_merge)),replace=False)
        X_train = np.vstack((X_train,gates_train_not_merge[train_indices,:]))

        test_indices = np.array(list(set(range(len(gates_train_not_merge))).difference(train_indices)))

        gates_not_merge_test = gates_train_not_merge[test_indices,:]

        y_train = y_train + [0]*len(train_indices)

        X_test = gates_merge_test
        X_test = np.vstack((X_test,gates_not_merge_test))

        y_test = [1]*len(gates_merge_test)+[0]*len(gates_not_merge_test)

        fclass = sk.svm.LinearSVC(penalty='l1',dual=False,class_weight='balanced')
        fclass.fit(X_train,y_train)

        print('\nLayer '+str(lyr+1))

        # Positive Examples
        print('Positive Accuracy',round(fclass.score(gates_merge_test,[1]*len(gates_merge_test))*100,2))

        # Negative Examples
        print('Negative Accuracy',round(fclass.score(gates_not_merge_test,[0]*len(gates_not_merge_test))*100,2))

        # Overall Accuracy
        print('All Accuracy',round(fclass.score(X_test,y_test)*100,2))

        if args.controls:
            # np.where(freg.coef_ != 0)
            shuffled_targets = merge_target.copy()
            np.random.shuffle(shuffled_targets)
            fclass_shuffled = sk.svm.LinearSVC(penalty='l1',dual=False,class_weight='balanced')
            fclass_shuffled.fit(gates_train,shuffled_targets)

            print('Shuffled')
            # Positive Examples
            print('Positive Accuracy',round(fclass_shuffled.score(gates_train_merge,[1]*len(gates_train_merge))*100,2))

            # Negative Examples
            print('Negative Accuracy',round(fclass_shuffled.score(gates_train_not_merge,[0]*len(gates_train_not_merge))*100,2))

            # Overall Accuracy
            print('All Accuracy',round(fclass_shuffled.score(gates_train,merge_target)*100,2))

            # Shuffled Accuracy
            print('Shuffled Accuracy',round(fclass_shuffled.score(gates_train,shuffled_targets)*100,2))

            random_tags = np.random.uniform(size=len(gates_train))
            random_tags = random_tags >= 0.5

            fclass_rand = sk.svm.LinearSVC(penalty='l1',dual=False,class_weight='balanced')
            fclass_rand.fit(gates_train,random_tags)

            print('Random')
            # Positive Examples
            print('Positive Accuracy',round(fclass_rand.score(gates_train_merge,[1]*len(gates_train_merge))*100,2))

            # Negative Examples
            print('Negative Accuracy',round(fclass_rand.score(gates_train_not_merge,[0]*len(gates_train_not_merge))*100,2))

            # Overall Accuracy
            print('All Accuracy',round(fclass_rand.score(gates_train,merge_target)*100,2))

            # Shuffled Accuracy
            print('Random Accuracy',round(fclass_rand.score(gates_train,random_tags)*100,2))
colors = ['#00a8ff','#9c88ff','#fbc531','#4cd137','#487eb0','#e84118','#7f8fa6','#273c75','#353b48','#FDA7DF','#833471']
mean_values = {}
for lyr in [0,1,2]:
    for delta in gate_activity_per_layer[lyr].keys():
        del_lyr = np.array(gate_activity_per_layer[lyr][delta])

        del_lyr = (del_lyr - mean_per_layer[lyr])/std_per_layer[lyr]

        mean_delta_value = del_lyr.mean()

        if delta in mean_values:
            mean_values[delta].append(mean_delta_value)
        else:
            mean_values[delta] = []
            mean_values[delta].append(mean_delta_value)
min_del = min(mean_values.keys())
for delta in mean_values.keys():
    plt.plot([1,2,3],mean_values[delta],label='Delta '+str(delta),markersize=10,color=colors[delta-min_del])
plt.xlabel('Layer')
plt.ylabel('Z-Scored Average Activation')
plt.title("Activation Change Over Layer in "+super_title)
# plt.set_xticklabels([1,2,3])
plt.legend()
plt.show()
plt.close()


        # significant_units = np.where(fclass.coef_ != 0)[1]
        #
        # significant_coefs = fclass.coef_[:,significant_units]
        # print('Number ',significant_coefs.shape[0]*significant_coefs.shape[1])
        # print('Sum ',np.sum(significant_coefs))
        #
        # z_score_gates = z_score_gates[:,significant_units]
        #
        # plt.imshow(z_score_gates, aspect='auto',cmap='seismic')
        # plt.clim(-np.max(z_score_gates), np.max(z_score_gates))
        # plt.colorbar()
        # # plt.title()
        # plt.show()



        # import pdb; pdb.set_trace()
    # plt.colorbar(vmin=0,vmax=1)
# Observations - it seems that this did work and there is some sort of statistical difference between vectors that are in the merged and non merged sets. The question now is how to measure this/visualize this difference... How do I viusalize the dynamics?

# the follow up question is: WE look atm at gates after the phrase ends and not at the last word of the phrase - maybe there is some significant information ther ethat we should look at...

# Could try maybe a kmeans approach? We state that there is exactly 2 different clusters of forget gates. and then we compare the sets of vectors? Could be interesting to see what it returns...
