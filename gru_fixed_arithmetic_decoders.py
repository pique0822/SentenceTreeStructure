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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


dataset_training = 'datasets/arithmetic/fixed_1e2/training.txt'
dataset_testing = 'datasets/arithmetic/fixed_1e2/testing.txt'

decoder_training_percent = 0.9

dataset = Dataset(dataset_training,dataset_testing)

num_layers = 1
hidden_size = 100
num_epochs = 449
input_size = dataset.vector_size
PATH = 'models/arithmetic_1e2_fixed_l_'+str(num_layers)+'_h_'+str(hidden_size)+'_ep_'+str(num_epochs)

model = GatedGRU(dataset.vector_size,hidden_size,output_size=1)
model.load_state_dict(torch.load(PATH))
model.eval()

temporal_hidden = {0:[],1:[],2:[],3:[],4:[],
                     5:[],6:[],7:[],8:[],9:[]}
by_line_first_num = []
by_line_second_num = []

for idx in range(dataset.testing_size()):
    running_sum = []
    partial_num = []

    input, label, line = dataset.testing_item(idx)

    addition_index = 4
    equals_index = 9

    first_num = line[:addition_index]
    second_num = line[addition_index+1:equals_index]

    true_first_num = int(first_num)
    true_second_num = int(second_num)

    x = torch.Tensor(input).reshape(1, -1, dataset.vector_size)

    hidden = model.init_hidden()

    decoded, hidden, (update_gates, reset_gates, hidden_states) = model(x,hidden)

    prediction = decoded[len(decoded)-1].item()

    by_line_first_num.append(true_first_num)
    by_line_second_num.append(true_second_num)

    for char_idx in range(len(line)):
        temporal_hidden[char_idx].append(hidden_states[char_idx].detach().numpy().reshape(-1))

training_idcs = np.random.choice(range(dataset.testing_size()),dataset.testing_size(),replace=False)

testing_idcs = training_idcs[int(dataset.testing_size()*decoder_training_percent):]
training_idcs = training_idcs[:int(dataset.testing_size()*decoder_training_percent)]

first_num_coefs = []
print('First Num')
for time in temporal_hidden.keys():
    regr = LinearRegression()
    X_train = np.array(temporal_hidden[time])[training_idcs,:]
    y_train = np.array(by_line_first_num).reshape(-1,1)[training_idcs,:]

    X_test = np.array(temporal_hidden[time])[testing_idcs,:]
    y_test = np.array(by_line_first_num).reshape(-1,1)[testing_idcs,:]

    regr.fit(X_train,y_train)

    y_pred = regr.predict(X_test)


    coef = np.log(regr.coef_.reshape(10,10))
    plt.title("Time: "+str(time+1)+" Linear Regression Model on First Number\nCoefficients Colored White to Black\nTest MSE = "+str(mean_squared_error(y_test, y_pred)))
    plt.imshow(coef,cmap='binary')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('linreg_first_num_coefs_time_'+str(time+1)+'.png')
    plt.close()

    first_num_coefs.append(coef)


second_num_coefs = []
print('Second Num')
for time in temporal_hidden.keys():
    regr = LinearRegression()
    X_train = np.array(temporal_hidden[time])[training_idcs,:]
    y_train = np.array(by_line_second_num).reshape(-1,1)[training_idcs,:]

    X_test = np.array(temporal_hidden[time])[testing_idcs,:]
    y_test = np.array(by_line_second_num).reshape(-1,1)[testing_idcs,:]

    regr.fit(X_train,y_train)

    y_pred = regr.predict(X_test)

    coef = np.log(regr.coef_.reshape(10,10))
    plt.title("Time: "+str(time+1)+" Linear Regression Model on Second Number\nCoefficients Colored White to Black\nTest MSE = "+str(mean_squared_error(y_test, y_pred)))
    plt.imshow(coef,cmap='binary')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('linreg_second_num_coefs_time_'+str(time+1)+'.png')
    plt.close()

    second_num_coefs.append(coef)

for time in range(len(second_num_coefs)):
    plt.title("Time: "+str(time+1)+"\nLinear Regression Normalized Coefficients Summed")
    coef1 = (first_num_coefs[time] - np.min(first_num_coefs[time]))/np.max(first_num_coefs[time] - np.min(first_num_coefs[time]))

    coef2 = (second_num_coefs[time] - np.min(second_num_coefs[time]))/np.max(second_num_coefs[time] - np.min(second_num_coefs[time]))

    plt.imshow(coef1 + coef2,cmap='binary', vmin=0,vmax=2)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
