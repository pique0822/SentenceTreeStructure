import torch.nn as nn
import torch

from Gated_GRU import GatedGRU
from datasets.arithmetic.arithmetic_dataset import Dataset

import numpy as np
import random

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

plot_intermediate = False
plot_running_sum = False
plot_final = False
plot_first_second = False
plot_pos_neg = False
plot_char = False
plot_category = False
plot_by_first_num = False
plot_by_second_num = False
plot_initial_paths = False
plot_by_MSE = False
study_clusters = False
plot_fully_injested = True

dim3 = False


dataset_training = 'datasets/arithmetic/1e2/training.txt'
dataset_testing = 'datasets/arithmetic/1e2/testing.txt'

dataset = Dataset(dataset_training,dataset_testing)

num_layers = 1
hidden_size = 100
num_epochs = 5
input_size = dataset.vector_size
PATH = 'models/arithmetic_l_'+str(num_layers)+'_h_'+str(hidden_size)+'_ep_'+str(num_epochs)

model = GatedGRU(dataset.vector_size,hidden_size,output_size=1)
model.load_state_dict(torch.load(PATH))
model.eval()
# average over all datasets

all_ugates = []
all_rgates = []
all_hidden = []

location = []

character = []

ordering = []

positive_negative = []
first_num = []

first_num_value = []
second_num_value = []

largest_abs_prediction = 0

predictions = []

mse = []

intermediate_predictions = []

running_sum = []

fully_injested_one = []
fully_injested_two = []

all_lines = []
for idx in range(dataset.testing_size()):
    relevant_order = []
    input, label, line = dataset.testing_item(idx)

    character.extend(np.argmax(np.array(input),1).tolist())

    addition_index = line.index('+')
    equals_index = line.index('=')

    first_num_pos = int(line[:addition_index]) >= 0
    second_num_pos = int(line[addition_index+1:equals_index]) >= 0

    x = torch.Tensor(input).reshape(1, -1, dataset.vector_size)

    hidden = model.init_hidden()

    decoded, hidden, (update_gates, reset_gates, hidden_states) = model(x,hidden)

    prediction = decoded[len(decoded)-1].item()

    previous_sum = 0
    for char_idx in range(len(line)):
        sub_line = line[:char_idx+1]

        try:
            parsum = eval(sub_line)
            previous_sum = parsum
        except:
            pass

        running_sum.append(previous_sum)

    if abs(prediction) > largest_abs_prediction:
        largest_abs_prediction = abs(prediction)

    for i in range(len(update_gates)):

        all_lines.append(line[:i])

        if i == addition_index:
            fully_injested_one.append(int(line[:addition_index]))
        else:
            fully_injested_one.append(0)

        if i == equals_index:
            fully_injested_two.append(int(line[addition_index+1:equals_index]))
        else:
            fully_injested_two.append(0)




        first_num_value.append(int(line[:addition_index]))
        second_num_value.append(int(line[addition_index+1:equals_index]))

        predictions.append(prediction)

        mse.append((prediction - label)**2)

        intermediate_predictions.append(decoded[i].item())

        relevant_order.append(idx+i)
        all_ugates.append(update_gates[i][0][0].detach().numpy())
        all_rgates.append(reset_gates[i][0][0].detach().numpy())
        all_hidden.append(hidden_states[i][0][0].detach().numpy())

        if i < addition_index:
            first_num.append(1)
            if first_num_pos:
                positive_negative.append(1)
            else:
                positive_negative.append(0)
            modification_index = 0
            location.append(modification_index)

        elif i == addition_index:
            first_num.append(0)
            positive_negative.append(4)
            modification_index = 1
            location.append(modification_index)

        elif addition_index < i < equals_index:
            first_num.append(2)
            if second_num_pos:
                positive_negative.append(3)
            else:
                positive_negative.append(2)
            modification_index = 2
            location.append(modification_index)

        elif i == equals_index:
            first_num.append(0)
            positive_negative.append(4)
            modification_index = 3
            location.append(modification_index)

    ordering.append(relevant_order)


color_dict = {0:'#9b59b6',1:'#3498db',2:'#f39c12',3:'#e74c3c'}
locations = []

# update gate
all_ugates = np.array(all_ugates)

# reset Gate
all_rgates = np.array(all_rgates)

# Output Gate
all_hidden = np.array(all_hidden)

pca_hidden = PCA(n_components=min(hidden_size,3))
pcah = pca_hidden.fit_transform(all_hidden)
pcah_var = np.sum(pca_hidden.explained_variance_ratio_)

pca_update = PCA(n_components=min(hidden_size,3))
pcau = pca_update.fit_transform(all_ugates)
pcau_var = np.sum(pca_update.explained_variance_ratio_)

pca_reset = PCA(n_components=min(hidden_size,3))
pcar = pca_reset.fit_transform(all_rgates)
pcar_var = np.sum(pca_reset.explained_variance_ratio_)


xmax_r = np.max(pcar[:,0])+0.5
xmax_h = np.max(pcah[:,0])+0.5
xmax_u = np.max(pcau[:,0])+0.5

xmin_r = np.min(pcar[:,0])-0.5
xmin_h = np.min(pcah[:,0])-0.5
xmin_u = np.min(pcau[:,0])-0.5

ymax_r = np.max(pcar[:,1])+0.5
ymax_h = np.max(pcah[:,1])+0.5
ymax_u = np.max(pcau[:,1])+0.5

ymin_r = np.min(pcar[:,1])-0.5
ymin_h = np.min(pcah[:,1])-0.5
ymin_u = np.min(pcau[:,1])-0.5

#
if plot_category:
    color_list = []
    for gate_idx in range(len(all_hidden)):
        color_list.append(color_dict[location[gate_idx]])


    relevant_idx = 0
    for idx, order in enumerate(ordering):
        if idx == relevant_idx:
            plt.plot(pcah[order,0],pcah[order,1],color='#000000', alpha=1)
        else:
            plt.plot(pcah[order,0],pcah[order,1],color='#000000', alpha=0)
    fig = plt.figure()
    plt.scatter(pcah[:,0],pcah[:,1],color=color_list)
    plt.title('Hidden State Viewing Location\n(Explained Variance : '+str(pcah_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()


    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)




    for idx, order in enumerate(ordering):
        if idx == relevant_idx:
            plt.plot(pcau[order,0],pcau[order,1],color='#000000', alpha=1)
        else:
            plt.plot(pcau[order,0],pcau[order,1],color='#000000', alpha=0)
    fig = plt.figure()
    plt.scatter(pcau[:,0],pcau[:,1],color=color_list)
    plt.title('Update Gate Viewing Location\n(Explained Variance : '+str(pcau_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)




    for idx, order in enumerate(ordering):
        if idx == relevant_idx:
            plt.plot(pcar[order,0],pcar[order,1],color='#000000', alpha=1)
        else:
            plt.plot(pcar[order,0],pcar[order,1],color='#000000', alpha=0)
    fig = plt.figure()
    plt.scatter(pcar[:,0],pcar[:,1],color=color_list)
    plt.title('Reset Gate Viewing Location\n(Explained Variance : '+str(pcar_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)



if plot_fully_injested:
    fully_injested_one = np.array(fully_injested_one)
    injested_idcs_one = np.argwhere(fully_injested_one).reshape(-1)
    first_colors = fully_injested_one[injested_idcs_one]



    fully_injested_two = np.array(fully_injested_two)
    injested_idcs_two = np.argwhere(fully_injested_two).reshape(-1)
    second_colors = fully_injested_two[injested_idcs_two]

    max_value = max(np.max(np.abs(fully_injested_one)), np.max(np.abs(fully_injested_two)))


    fig = plt.figure()
    plt.scatter(pcah[injested_idcs_one,0],pcah[injested_idcs_one,1],c=first_colors, cmap='seismic',vmin=-max_value, vmax=max_value)
    plt.title('Hidden State\nIngest First Num Colored by Ingestion\n(Explained Variance : '+str(pcah_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim(xmin_h,xmax_h)
    plt.ylim(ymin_h,ymax_h)
    plt.colorbar()
    plt.tight_layout()

    fig = plt.figure()
    plt.scatter(pcah[injested_idcs_two,0],pcah[injested_idcs_two,1],c=second_colors, cmap='seismic',vmin=-max_value, vmax=max_value)
    plt.title('Hidden State\nIngest Second Num Colored by Ingestion\n(Explained Variance : '+str(pcah_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim(xmin_h,xmax_h)
    plt.ylim(ymin_h,ymax_h)
    plt.colorbar()
    plt.tight_layout()



    fig = plt.figure()
    plt.scatter(pcau[injested_idcs_one,0],pcau[injested_idcs_one,1],c=first_colors, cmap='seismic',vmin=-max_value, vmax=max_value)
    plt.title('Update Gate\nIngest First Num Colored by Ingestion\n(Explained Variance : '+str(pcau_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim(xmin_u,xmax_u)
    plt.ylim(ymin_u,ymax_u)
    plt.colorbar()
    plt.tight_layout()

    fig = plt.figure()
    plt.scatter(pcau[injested_idcs_two,0],pcau[injested_idcs_two,1],c=second_colors, cmap='seismic',vmin=-max_value, vmax=max_value)
    plt.title('Update Gate\nIngest Second Num Colored by Ingestion\n(Explained Variance : '+str(pcau_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim(xmin_u,xmax_u)
    plt.ylim(ymin_u,ymax_u)
    plt.colorbar()
    plt.tight_layout()




    fig = plt.figure()
    plt.scatter(pcar[injested_idcs_one,0],pcar[injested_idcs_one,1],c=first_colors, cmap='seismic',vmin=-max_value, vmax=max_value)
    plt.title('Reset Gate\nIngest First Num Colored by Ingestion\n(Explained Variance : '+str(pcar_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim(xmin_r,xmax_r)
    plt.ylim(ymin_r,ymax_r)
    plt.colorbar()
    plt.tight_layout()

    fig = plt.figure()
    plt.scatter(pcar[injested_idcs_two,0],pcar[injested_idcs_two,1],c=second_colors, cmap='seismic',vmin=-max_value, vmax=max_value)
    plt.title('Reset Gate\nIngest Second Num Colored by Ingestion\n(Explained Variance : '+str(pcar_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim(xmin_r,xmax_r)
    plt.ylim(ymin_r,ymax_r)
    plt.colorbar()
    plt.tight_layout()



if plot_char:
    color_dict = {0:'#c56cf0',1:'#ffb8b8',2:'#ff3838',3:'#ff9f1a',4:'#fff200',5:'#3ae374',6:'#67e6dc',7:'#17c0eb',8:'#7158e2',9:'#3d3d3d',10:'#0984e3',11:'#fdcb6e',12:'#7f8c8d'}

    color_list = []
    for gate_idx in range(len(all_hidden)):
        color_list.append(color_dict[character[gate_idx]])
    fig = plt.figure()
    plt.scatter(pcah[:,0],pcah[:,1],color=color_list)
    plt.title('Hidden State Viewing Characters\n(Explained Variance : '+str(pcah_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)



    fig = plt.figure()
    plt.scatter(pcau[:,0],pcau[:,1],color=color_list)
    plt.title('Update Gate Viewing Characters\n(Explained Variance : '+str(pcau_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)



    fig = plt.figure()
    plt.scatter(pcar[:,0],pcar[:,1],color=color_list)
    plt.title('Reset Gate Viewing Characters\n(Explained Variance : '+str(pcar_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)




# visualize positive versus negative numbers
if plot_pos_neg:
    color_dict = {0:[0.08, 0.625, 0.52,1.0],1:[0.15, 0.68, 0.375,1.0],2:[0.82, 0.33, 0,1.0],3:[0.75, 0.22, 0.17,1.0],4:[0,0,0,0]}
    color_list = []
    for gate_idx in range(len(all_hidden)):
        color_list.append(color_dict[positive_negative[gate_idx]])

    fig = plt.figure()
    plt.scatter(pcah[:,0],pcah[:,1],color=color_list)
    plt.title('Hidden State Viewing Positive / Negative \n(Explained Variance : '+str(pcah_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)



    fig = plt.figure()
    plt.scatter(pcau[:,0],pcau[:,1],color=color_list)
    plt.title('Update Gate Viewing Positive / Negative \n(Explained Variance : '+str(pcau_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)



    fig = plt.figure()
    plt.scatter(pcar[:,0],pcar[:,1],color=color_list)
    plt.title('Reset Gate Viewing Positive / Negative\n(Explained Variance : '+str(pcar_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)




# colorgin by number
if plot_first_second:
    color_dict = {0:[0,0,0,0],1:[1,0,0,1],2:[0,0,1,1]}
    color_list = []
    for gate_idx in range(len(all_hidden)):
        color_list.append(color_dict[first_num[gate_idx]])

    fig = plt.figure()
    plt.scatter(pcah[:,0],pcah[:,1],color=color_list)
    plt.title('Hidden State Viewing Number Location\n(Explained Variance : '+str(pcah_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)



    fig = plt.figure()
    plt.scatter(pcau[:,0],pcau[:,1],color=color_list)
    plt.title('Update Gate Viewing Number Location\n(Explained Variance : '+str(pcau_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)



    fig = plt.figure()
    plt.scatter(pcar[:,0],pcar[:,1],color=color_list)
    plt.title('Reset Gate Viewing Number Location\n(Explained Variance : '+str(pcar_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints=1)


# coloring based on cluster
if study_clusters:
    # 1oo
    #5,-1.5,0
    # -2.25, -1, 1

    x_cen = 5
    y_cen = -1.5
    z_cen = 0

    x = np.argwhere(np.abs(pcar[:,0] - x_cen) <= 1)
    y = np.argwhere(np.abs(pcar[:,1] - y_cen) <= 1)
    z = np.argwhere(np.abs(pcar[:,2] - z_cen) <= 1)

    idcs = set(x.reshape(-1).tolist())
    idcs = idcs.intersection(set(y.reshape(-1).tolist()))
    idcs = idcs.intersection(set(z.reshape(-1).tolist()))

    idcs = np.array(list(idcs))

    all_lines = np.array(all_lines)
    # print(all_lines[idcs])

    predictions = np.array(predictions)

    fig = plt.figure()
    plt.scatter(pcah[:,0],pcah[:,1],c=['#FFFFFF']*len(predictions), cmap='seismic',vmin=-largest_abs_prediction, vmax=largest_abs_prediction)
    plt.scatter(pcah[idcs,0],pcah[idcs,1],c=predictions[idcs], cmap='seismic',vmin=-largest_abs_prediction, vmax=largest_abs_prediction)
    plt.title('Hidden State\nColored by Final Prediction\n(Explained Variance : '+str(pcah_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


    fig = plt.figure()
    plt.scatter(pcau[:,0],pcau[:,1],c=['#FFFFFF']*len(predictions), cmap='seismic',vmin=-largest_abs_prediction, vmax=largest_abs_prediction)
    plt.scatter(pcau[idcs,0],pcau[idcs,1],c=predictions[idcs], cmap='seismic',vmin=-largest_abs_prediction, vmax=largest_abs_prediction)
    plt.title('Update Gate\nColored by Final Prediction\n(Explained Variance : '+str(pcau_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


    fig = plt.figure()
    plt.scatter(pcar[:,0],pcar[:,1],c=['#FFFFFF']*len(predictions), cmap='seismic',vmin=-largest_abs_prediction, vmax=largest_abs_prediction)
    plt.scatter(pcar[idcs,0],pcar[idcs,1],c=predictions[idcs], cmap='seismic',vmin=-largest_abs_prediction, vmax=largest_abs_prediction)
    plt.title('Reset Gate\nColored by Final Prediction\n(Explained Variance : '+str(pcar_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()



# coloring based on final output
if plot_final:
    fig = plt.figure()
    plt.scatter(pcah[:,0],pcah[:,1],c=predictions, cmap='seismic',vmin=-largest_abs_prediction, vmax=largest_abs_prediction)
    plt.title('Hidden State\nColored by Final Prediction\n(Explained Variance : '+str(pcah_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


    fig = plt.figure()
    plt.scatter(pcau[:,0],pcau[:,1],c=predictions, cmap='seismic',vmin=-largest_abs_prediction, vmax=largest_abs_prediction)
    plt.title('Update Gate\nColored by Final Prediction\n(Explained Variance : '+str(pcau_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


    fig = plt.figure()
    plt.scatter(pcar[:,0],pcar[:,1],c=predictions, cmap='seismic',vmin=-largest_abs_prediction, vmax=largest_abs_prediction)
    plt.title('Reset Gate\nColored by Final Prediction\n(Explained Variance : '+str(pcar_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


if plot_running_sum:
    running_sum = np.array(running_sum)

    largest_sum = np.max(np.abs(running_sum))

    fig = plt.figure()
    plt.scatter(pcah[:,0],pcah[:,1],c=running_sum, cmap='seismic',vmin=-largest_sum, vmax=largest_sum)
    plt.title('Hidden State\nColored by Running Sum\n(Explained Variance : '+str(pcah_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


    fig = plt.figure()
    plt.scatter(pcau[:,0],pcau[:,1],c=running_sum, cmap='seismic',vmin=-largest_sum, vmax=largest_sum)
    plt.title('Update Gate\nColored by Running Sum\n(Explained Variance : '+str(pcau_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


    fig = plt.figure()
    plt.scatter(pcar[:,0],pcar[:,1],c=running_sum, cmap='seismic',vmin=-largest_sum, vmax=largest_sum)
    plt.title('Reset Gate\nColored by Running Sum\n(Explained Variance : '+str(pcar_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()



# coloring based on MSE
if plot_by_MSE:
    fig = plt.figure()
    plt.scatter(pcah[:,0],pcah[:,1],c=mse, cmap='Reds',vmin=0, vmax=np.max(np.abs(np.array(mse))))
    plt.title('Hidden State\nColored by MSE\n(Explained Variance : '+str(pcah_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


    fig = plt.figure()
    plt.scatter(pcau[:,0],pcau[:,1],c=mse, cmap='Reds',vmin=0, vmax=np.max(np.abs(np.array(mse))))
    plt.title('Update Gate\nColored by MSE\n(Explained Variance : '+str(pcau_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


    fig = plt.figure()
    plt.scatter(pcar[:,0],pcar[:,1],c=mse, cmap='Reds',vmin=0, vmax=np.max(np.abs(np.array(mse))))
    plt.title('Reset Gate\nColored by MSE\n(Explained Variance : '+str(pcar_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


if plot_by_first_num:
    fig = plt.figure()

    largest_value = max(abs(np.array(first_num_value)))
    plt.scatter(pcah[:,0],pcah[:,1],c=first_num_value, cmap='seismic',vmin=-largest_value, vmax=largest_value)
    plt.title('Hidden State\nColored by First Number Value\n(Explained Variance : '+str(pcah_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()
    #
    #

    fig = plt.figure()
    plt.scatter(pcau[:,0],pcau[:,1],c=first_num_value, cmap='seismic',vmin=-largest_value, vmax=largest_value)
    plt.title('Update Gate\nColored by First Number\n(Explained Variance : '+str(pcau_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()



    fig = plt.figure()
    plt.scatter(pcar[:,0],pcar[:,1],c=first_num_value, cmap='seismic',vmin=-largest_value, vmax=largest_value)
    plt.title('Reset Gate\nColored by First Number\n(Explained Variance : '+str(pcar_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


if plot_by_second_num:
    fig = plt.figure()
    largest_value = max(abs(np.array(second_num_value)))
    plt.scatter(pcah[:,0],pcah[:,1],c=second_num_value, cmap='seismic',vmin=-largest_value, vmax=largest_value)
    plt.title('Hidden State\nColored by Second Number Value\n(Explained Variance : '+str(pcah_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


    fig = plt.figure()
    plt.scatter(pcau[:,0],pcau[:,1],c=second_num_value, cmap='seismic',vmin=-largest_value, vmax=largest_value)
    plt.title('Update Gate\nColored by Second Number Value\n(Explained Variance : '+str(pcau_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()


    fig = plt.figure()
    plt.scatter(pcar[:,0],pcar[:,1],c=second_num_value, cmap='seismic',vmin=-largest_value, vmax=largest_value)
    plt.title('Reset Gate\nColored by Second Number Value\n(Explained Variance : '+str(pcar_var)+')')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.tight_layout()



# coloring based on intermediate output
if plot_intermediate:
    largest_int_pred = np.max(np.abs(np.array(intermediate_predictions)))

    if dim3:
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcah[:,0],pcah[:,1],pcah[:,2],c=intermediate_predictions, cmap='seismic',vmin=-largest_int_pred, vmax=largest_int_pred)

        plt.title('Hidden State\nColored by Intermediate Prediction\n(Explained Variance : '+str(pcah_var)+')')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.tight_layout()


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcau[:,0],pcau[:,1],pcau[:,2],c=intermediate_predictions, cmap='seismic',vmin=-largest_int_pred, vmax=largest_int_pred)

        plt.title('Update Gate\nColored by Intermediate Prediction\n(Explained Variance : '+str(pcau_var)+')')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.tight_layout()



        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcar[:,0],pcar[:,1],pcar[:,2],c=intermediate_predictions, cmap='seismic',vmin=-largest_int_pred, vmax=largest_int_pred)

        plt.title('Reset Gate\nColored by Intermediate Prediction\n(Explained Variance : '+str(pcar_var)+')')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.tight_layout()
    else:
        fig = plt.figure()
        plt.scatter(pcah[:,0],pcah[:,1],c=intermediate_predictions, cmap='seismic',vmin=-largest_int_pred, vmax=largest_int_pred)
        plt.title('Hidden State\nColored by Intermediate Prediction\n(Explained Variance : '+str(pcah_var)+')')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar()
        plt.tight_layout()


        fig = plt.figure()
        plt.scatter(pcau[:,0],pcau[:,1],c=intermediate_predictions, cmap='seismic',vmin=-largest_int_pred, vmax=largest_int_pred)
        plt.title('Update Gate\nColored by Intermediate Prediction\n(Explained Variance : '+str(pcau_var)+')')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar()
        plt.tight_layout()


        fig = plt.figure()
        plt.scatter(pcar[:,0],pcar[:,1],c=intermediate_predictions, cmap='seismic',vmin=-largest_int_pred, vmax=largest_int_pred)
        plt.title('Reset Gate\nColored by Intermediate Prediction\n(Explained Variance : '+str(pcar_var)+')')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar()
        plt.tight_layout()



# coloring based on intermediate output
if plot_initial_paths:
    color_dict = {0:'#c56cf0',1:'#c56cf0',2:'#c56cf0',3:'#c56cf0',4:'#c56cf0',5:'#c56cf0',6:'#c56cf0',7:'#c56cf0',8:'#c56cf0',9:'#c56cf0',10:'#0984e3',11:'#0984e3',12:'#0984e3'}

    initial_hiddens = []
    initial_updates = []
    initial_resets = []

    colors = []
    for i in range(dataset.vector_size):
        char_vec = np.zeros(dataset.vector_size)
        char_vec[i] = 1

        colors.append(color_dict[i])

        x = torch.Tensor(char_vec).reshape(1, -1, dataset.vector_size)

        hidden = model.init_hidden()
        decoded, hidden, (update_gates, reset_gates, hidden_states) = model(x,hidden)

        initial_resets.append(reset_gates[0][0].detach().numpy())
        initial_hiddens.append(hidden[0].detach().numpy())
        initial_updates.append(update_gates[0][0].detach().numpy())

        char_vec = np.zeros(dataset.vector_size)
        char_vec[11] = 1

        x = torch.Tensor(char_vec).reshape(1, -1, dataset.vector_size)

        decoded, hidden, (update_gates, reset_gates, hidden_states) = model(x,hidden)

        char_vec = np.zeros(dataset.vector_size)
        char_vec[1] = 1

        x = torch.Tensor(char_vec).reshape(1, -1, dataset.vector_size)

        decoded, hidden, (update_gates, reset_gates, hidden_states) = model(x,hidden)

        char_vec = np.zeros(dataset.vector_size)
        char_vec[12] = 1

        x = torch.Tensor(char_vec).reshape(1, -1, dataset.vector_size)

        decoded, hidden, (update_gates, reset_gates, hidden_states) = model(x,hidden)

        print(str(i)+'+1=',decoded[0].item())

    initial_resets = np.array(initial_resets).reshape(-1,hidden_size)
    initial_hiddens = np.array(initial_hiddens).reshape(-1,hidden_size)
    initial_updates = np.array(initial_updates).reshape(-1,hidden_size)

    pcar = pca_reset.transform(initial_resets)
    pcau = pca_update.transform(initial_updates)
    pcah = pca_hidden.transform(initial_hiddens)

    fig = plt.figure()
    if hidden_size > 2:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcah[:,0],pcah[:,1],pcah[:,2],c=colors)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(pcah[:,0],pcah[:,1],c=colors)
    plt.title('Hidden State\nColored by Character\n(Explained Variance : '+str(pcah_var)+')')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if hidden_size > 2:
        ax.set_zlabel('PC3')
    # ax.colorbar()
    # ax.tight_layout()



    fig = plt.figure()
    if hidden_size > 2:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcau[:,0],pcau[:,1],pcau[:,2],c=colors)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(pcau[:,0],pcau[:,1],c=colors)

    plt.title('Update Gate\nColored by Character\n(Explained Variance : '+str(pcau_var)+')')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if hidden_size > 2:
        ax.set_zlabel('PC3')
    # plt.colorbar()
    # plt.tight_layout()



    fig = plt.figure()
    if hidden_size > 2:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcar[:,0],pcar[:,1],pcar[:,2],c=colors)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(pcar[:,0],pcar[:,1],c=colors)
    plt.title('Reset Gate\nColored by Character\n(Explained Variance : '+str(pcar_var)+')')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if hidden_size > 2:
        ax.set_zlabel('PC3')
    # plt.colorbar()
    plt.tight_layout()



plt.show()
plt.close()
