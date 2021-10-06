import sys 
import tensorflow as tf
import matplotlib.pyplot as plt 
import scipy.io 
import numpy as np 
from scipy.interpolate import griddata 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time 
from pinn_models import DiNucciPINNFlow, DiNucciNondimPINNFlow
import logging
import os

def load_data(name, n, non_dimensionalise=True):
    """ loads dataset n"""
    data = scipy.io.loadmat("data/steady/%s_exp%d.mat" %(name, n)) 
    Q = data['Q'][0][0]
    K_truth = data['K'][0][0]
    
    x_data = data['xexp'][:,0]
    u_data = data['hexp']
    # print(x_data)
    L = data['L'][0][0]
    W = data['W'][0][0]

    # x_data = L - x_data 
    q_data = -np.ones(x_data.shape) * Q/W 

    hc = np.sqrt(-q_data[0] * L / K_truth)

    if non_dimensionalise:
        """ non dimensonalise by critical """
        x_data /= L
        u_data /= hc
        q_data /= L*K_truth
    X_data = np.stack((x_data, q_data)).T

    return X_data, u_data, L, W, K_truth 

def load_all(name, n_max):
    """ load all training data into a dictionary 
    stored in order of X, u, L, W, k""" 
    training_data = dict() 
    for i in range(n_max):
        training_data[i+1] = load_data(name, i+1) 

    return training_data 

def make_training_set(ind_list, training_data):
    """ compile the training set corresponding
    to experiments listed in ind_list """ 
    
    exp = training_data[ind_list[0]] 
    X_train = exp[0]
    u_train = exp[1] 

    for i in ind_list[1:]: 
        exp = training_data[i]
        X_train = np.append(X_train, exp[0], axis=0)
        u_train = np.append(u_train, exp[1], axis=0)

    return X_train, u_train 

def viz_data_nondim(ind_list, training_data, color=None):
    """ visualize all of the training data on the same axis """
    for ind in ind_list:
        X, u, L, W, k = training_data[ind]
        x = X[:,0]
        if color is None:
            plt.plot(x, u, 'ob');
        else:
            format_str = 'o' + color 
            plt.plot(x, u, format_str)

    plt.grid(True)
    plt.xlim([0,1])
    plt.ylim([0,None])
    plt.xlabel("x'")
    plt.ylabel("h'")
        


# Define dataset name 
data_name = "steady_state_2mm_nolake"
n_max = 24
n_train = 24

# Exract training data 
training_data = load_all(data_name, n_max) 
training_list = [i+1 for i in range(n_train)]
viz_data_nondim(training_list, training_data, 'r')

data_name = "steady_state_1mm_nolake"
n_max = 26
n_train = 26

# Exract training data 
training_data = load_all(data_name, n_max) 
training_list = [i+1 for i in range(n_train)]
viz_data_nondim(training_list, training_data, 'b')
plt.show()
