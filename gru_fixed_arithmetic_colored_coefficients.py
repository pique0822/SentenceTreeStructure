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

def function_scaling(x):
    return x
    # if x < 0:
    #     return -np.log(-x+1)
    # else:
    #     return np.log(x+1)


dataset_training = 'datasets/arithmetic/fixed_1e2/training.txt'
dataset_testing = 'datasets/arithmetic/fixed_1e2/testing.txt'

decoder_training_percent = 0.9

dataset = Dataset(dataset_training,dataset_testing)

num_layers = 1
hidden_size = 100
num_epochs = 449
input_size = dataset.vector_size
PATH = 'models/arithmetic_1e2_fixed_l_'+str(num_layers)+'_h_'+str(hidden_size)+'_ep_'+str(num_epochs)

regression_trials = 1

model = GatedGRU(dataset.vector_size,hidden_size,output_size=1)
model.load_state_dict(torch.load(PATH))
model.eval()

temporal_hidden = {0:[],1:[],2:[],3:[],4:[],
                     5:[],6:[],7:[],8:[],9:[]}
temporal_categorical_running_sum = {0:[],1:[],2:[],3:[],4:[],
                     5:[],6:[],7:[],8:[],9:[]}
temporal_incremental_running_sum = {0:[],1:[],2:[],3:[],4:[],
                     5:[],6:[],7:[],8:[],9:[]}
by_line_first_num = []
by_line_second_num = []

for idx in range(dataset.testing_size()):
    running_sum = []
    categorical_sum = [0]*10
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
        sub_line = line[:char_idx+1]
        num_one = 0
        num_two = 0
        if len(sub_line) <= 5:
            try:
                num_one = int(line[:min(char_idx+1,4)])
            except:
                num_one = 0
        if len(sub_line) > 5:
            num_one = int(line[:4])
            try:
                num_two = int(line[5:min(char_idx+1,9)])
            except:
                num_two = 0

        running_sum.append(num_one + num_two)


    categorical_sum[0] = 0
    categorical_sum[1] = int(true_first_num / 100)*100
    categorical_sum[2] = int(true_first_num / 10)*10
    categorical_sum[3] = true_first_num
    categorical_sum[4] = categorical_sum[3]
    categorical_sum[5] = categorical_sum[3]
    categorical_sum[6] = true_first_num + (int(true_second_num / 100) * 100)
    categorical_sum[7] = true_first_num + (int(true_second_num / 10) * 10)
    categorical_sum[8] = true_first_num + true_second_num
    categorical_sum[9] = true_first_num + true_second_num

    for char_idx in range(len(line)):
        temporal_hidden[char_idx].append(hidden_states[char_idx].detach().numpy().reshape(-1))
        temporal_incremental_running_sum[char_idx].append(running_sum[char_idx])
        temporal_categorical_running_sum[char_idx].append(categorical_sum[char_idx])
folder = 'fixed_arithmetic_results/hidden/'


first_num_coefs = []
print('First Num')
for time in temporal_hidden.keys():
    best_reg = None
    best_test_MSE = np.inf
    best_train_MSE = np.inf
    for trial in range(regression_trials):

        training_idcs = np.random.choice(range(dataset.testing_size()),dataset.testing_size(),replace=False)
        testing_idcs = training_idcs[int(dataset.testing_size()*decoder_training_percent):]
        training_idcs = training_idcs[:int(dataset.testing_size()*decoder_training_percent)]

        regr = LinearRegression()
        X_train = np.array(temporal_hidden[time])[training_idcs,:]
        y_train = np.array(by_line_first_num).reshape(-1,1)[training_idcs,:]

        X_test = np.array(temporal_hidden[time])[testing_idcs,:]
        y_test = np.array(by_line_first_num).reshape(-1,1)[testing_idcs,:]

        regr.fit(X_train,y_train)

        y_test_pred = regr.predict(X_test)
        y_train_pred = regr.predict(X_train)

        if mean_squared_error(y_test, y_test_pred) < best_test_MSE:
            best_test_MSE = mean_squared_error(y_test, y_test_pred)
            best_train_MSE = mean_squared_error(y_train, y_train_pred)
            best_reg = regr




    reshaped_coefs = best_reg.coef_.reshape(10,10)
    coef = np.zeros((10,10))
    for row in range(10):
        for col in range(10):
            coef[row,col] = function_scaling(reshaped_coefs[row,col])

    plt.title("Time: "+str(time+1)+" Linear Regression Model on First Number\nCoefficients Colored White to Black\nTrain MSE = "+str(best_train_MSE)+"\nTest MSE = "+str(best_test_MSE))
    plt.imshow(coef,cmap='binary')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(folder+'linreg_first_num_coefs_time_'+str(time+1)+'.png')
    plt.close()

    first_num_coefs.append(coef)

second_num_coefs = []
print('Second Num')
for time in temporal_hidden.keys():
    best_reg = None
    best_test_MSE = np.inf
    best_train_MSE = np.inf
    for trial in range(regression_trials):

        training_idcs = np.random.choice(range(dataset.testing_size()),dataset.testing_size(),replace=False)
        testing_idcs = training_idcs[int(dataset.testing_size()*decoder_training_percent):]
        training_idcs = training_idcs[:int(dataset.testing_size()*decoder_training_percent)]


        regr = LinearRegression()
        X_train = np.array(temporal_hidden[time])[training_idcs,:]
        y_train = np.array(by_line_second_num).reshape(-1,1)[training_idcs,:]

        X_test = np.array(temporal_hidden[time])[testing_idcs,:]
        y_test = np.array(by_line_second_num).reshape(-1,1)[testing_idcs,:]

        regr.fit(X_train,y_train)

        y_test_pred = regr.predict(X_test)
        y_train_pred = regr.predict(X_train)

        if mean_squared_error(y_test, y_test_pred) < best_test_MSE:
            best_test_MSE = mean_squared_error(y_test, y_test_pred)
            best_train_MSE = mean_squared_error(y_train, y_train_pred)
            best_reg = regr

    reshaped_coefs = best_reg.coef_.reshape(10,10)
    coef = np.zeros((10,10))
    for row in range(10):
        for col in range(10):
            coef[row,col] = function_scaling(reshaped_coefs[row,col])

    plt.title("Time: "+str(time+1)+" Linear Regression Model on Second Number\nCoefficients Colored White to Black\nTrain MSE = "+str(best_train_MSE)+"\nTest MSE = "+str(best_test_MSE))
    plt.imshow(coef,cmap='binary')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(folder+'linreg_second_num_coefs_time_'+str(time+1)+'.png')
    plt.close()

    second_num_coefs.append(coef)



incremental_running_sum_coefs = []
print('Incremental Running Sum')
for time in temporal_hidden.keys():
    best_reg = None
    best_test_MSE = np.inf
    best_train_MSE = np.inf
    for trial in range(regression_trials):

        training_idcs = np.random.choice(range(dataset.testing_size()),dataset.testing_size(),replace=False)
        testing_idcs = training_idcs[int(dataset.testing_size()*decoder_training_percent):]
        training_idcs = training_idcs[:int(dataset.testing_size()*decoder_training_percent)]


        regr = LinearRegression()
        X_train = np.array(temporal_hidden[time])[training_idcs,:]
        y_train = np.array(temporal_incremental_running_sum[time]).reshape(-1,1)[training_idcs,:]

        X_test = np.array(temporal_hidden[time])[testing_idcs,:]
        y_test = np.array(temporal_incremental_running_sum[time]).reshape(-1,1)[testing_idcs,:]

        regr.fit(X_train,y_train)

        y_test_pred = regr.predict(X_test)
        y_train_pred = regr.predict(X_train)

        if mean_squared_error(y_test, y_test_pred) < best_test_MSE:
            best_test_MSE = mean_squared_error(y_test, y_test_pred)
            best_train_MSE = mean_squared_error(y_train, y_train_pred)
            best_reg = regr

    reshaped_coefs = best_reg.coef_.reshape(10,10)
    coef = np.zeros((10,10))
    for row in range(10):
        for col in range(10):
            coef[row,col] = function_scaling(reshaped_coefs[row,col])

    plt.title("Time: "+str(time+1)+" Linear Regression Model on Incremental Running Sum\nCoefficients Colored White to Black\nTrain MSE = "+str(best_train_MSE)+"\nTest MSE = "+str(best_test_MSE))
    plt.imshow(coef,cmap='binary')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(folder+'linreg_incremental_sum_coefs_time_'+str(time+1)+'.png')
    plt.close()

    incremental_running_sum_coefs.append(coef)






categorical_running_sum_coefs = []
print('Categorical Running Sum')
for time in temporal_hidden.keys():
    best_reg = None
    best_test_MSE = np.inf
    best_train_MSE = np.inf
    for trial in range(regression_trials):

        training_idcs = np.random.choice(range(dataset.testing_size()),dataset.testing_size(),replace=False)
        testing_idcs = training_idcs[int(dataset.testing_size()*decoder_training_percent):]
        training_idcs = training_idcs[:int(dataset.testing_size()*decoder_training_percent)]


        regr = LinearRegression()
        X_train = np.array(temporal_hidden[time])[training_idcs,:]
        y_train = np.array(temporal_categorical_running_sum[time]).reshape(-1,1)[training_idcs,:]

        X_test = np.array(temporal_hidden[time])[testing_idcs,:]
        y_test = np.array(temporal_categorical_running_sum[time]).reshape(-1,1)[testing_idcs,:]

        regr.fit(X_train,y_train)

        y_test_pred = regr.predict(X_test)
        y_train_pred = regr.predict(X_train)

        if mean_squared_error(y_test, y_test_pred) < best_test_MSE:
            best_test_MSE = mean_squared_error(y_test, y_test_pred)
            best_train_MSE = mean_squared_error(y_train, y_train_pred)
            best_reg = regr


    reshaped_coefs = best_reg.coef_.reshape(10,10)
    coef = np.zeros((10,10))
    for row in range(10):
        for col in range(10):
            coef[row,col] = function_scaling(reshaped_coefs[row,col])

    plt.title("Time: "+str(time+1)+" Linear Regression Model on Categorical Running Sum\nCoefficients Colored White to Black\nTrain MSE = "+str(best_train_MSE)+"\nTest MSE = "+str(best_test_MSE))
    plt.imshow(coef,cmap='binary')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(folder+'linreg_categorical_sum_coefs_time_'+str(time+1)+'.png')
    plt.close()

    incremental_running_sum_coefs.append(coef)
