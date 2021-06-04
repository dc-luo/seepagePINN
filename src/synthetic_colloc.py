import sys 
import tensorflow as tf
import matplotlib.pyplot as plt 
import scipy.io 
import numpy as np 
from scipy.interpolate import griddata 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time 
import logging
import os
import h5py 
import argparse

from pinn_models import DiNucciNormalizedScaledPINNFlow, DupuitNormalizedScaledPINNFlow
from utils import *

# Tensorflow logging
os.system("mkdir steady_figures")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

FIGSIZE=(12,8)
plt.rcParams.update({'font.size' : 16})

######################################################################
# Parsing in PDE model as input 
parser = argparse.ArgumentParser(description='Select PDE model') 
parser.add_argument('-m', '--model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help="PDE choice: dinucci or dupuit")
parser.add_argument("-r", "--random", help="Do not set constant seed", action="store_true", default=False)
args = parser.parse_args()

######################################################################
if not args.random:
    # Set random seed
    np.random.seed(1234)
    tf.set_random_seed(1234) 

flow_model = args.model

print("Data generated using ", flow_model)
print("NN interpret using ", flow_model)
print("Random: ", args.random)

### Training data ###
L = 1.0 
K = 0.2 
h2 = 0
n_max = 15
n_train = n_max
n_points = 30

noise_ratio = 0.00

# Define training flow rates 
q_min = 0.1
q_max = 0.2
q_list = np.linspace(q_min, q_max, n_max)
scale_q = q_min

# Define training set 
training_list = [i+1 for i in range(n_train)]

# Refined grid points for testing the solution, including interpolation and extrapolation 
n_points_refined = 100
q_test = np.arange(0.5*q_min, 8.5*q_min, 0.5*q_min)
test_list = [i+1 for i in range(q_test.shape[0])]

if flow_model == "dinucci":
    # Convert to h1
    h1_list = np.sqrt(2*L*q_list/K + h2**2)
    noise_sd = noise_ratio * np.max(h1_list)

    # Make synethetic data 
    training_data = make_synthetic_all_dinucci(h1_list, h2, L, K, n_points, scale_q=scale_q, noise_sd=noise_sd)

    # Make test data
    h1_test = np.sqrt(2*L*q_test/K + h2**2)
    test_data = make_synthetic_all_dinucci(h1_test, h2, L, K, n_points_refined, scale_q=scale_q, noise_sd=0)

if flow_model == "dupuit":
    # First run with max q to determine an appropriate noise sd 
    h_max = dupuit_analytic(0, L, q_max, K, L)
    noise_sd = noise_ratio * h_max
    
    # Make synthetic dupuit data 
    training_data = make_synthetic_all_dupuit(q_list, h2, L, K, n_points, scale_q=scale_q, noise_sd=noise_sd)

    # Refined grid points at same training q values for plotting exact solution and PINN prediction
    test_data = make_synthetic_all_dupuit(q_test, h2, L, K, n_points_refined, scale_q=scale_q, noise_sd=0)

# Make training set 
X_train, u_train = make_training_set(training_list, training_data) 
X, u, L, W, k = training_data[1] 

# Make test set 
X_test, u_test = make_training_set(test_list, test_data)

# Train model and save solutions/data to hdf5 format
save_path = "paper/synthetic/collocation/"
out_file = h5py.File(save_path + "data_" + flow_model + ".h5", 'w')
out_file.create_dataset('q_data', data=q_list)
out_file.create_dataset('scale_q', data=scale_q)
out_file.create_dataset('X_data', data=X_train)
out_file.create_dataset('u_data', data=u_train)
out_file.create_dataset('q_test', data=q_test)
out_file.create_dataset('X_test', data=X_test)
out_file.create_dataset('u_test', data=u_test)
out_file.create_dataset('noise_ratio', data=noise_ratio)

######################################################################
# Neural network
N_train = 20000
layers = [2, 20, 20, 20, 20, 20, 1]
# alpha = 1.0

h_max_train = np.max(u_train[:,0])
alpha = 1 * h_max_train**2
q_colloc = np.linspace(q_max, 2*q_max, 30)
######################################################################
# 1. Train model using no collocation points
n_colloc = 0
X_colloc = None 
if flow_model == "dupuit":
    # dinucci flow model fitting K
    model = DupuitNormalizedScaledPINNFlow(X_train, u_train, k, layers, 0, 1, scale_q=scale_q, 
            X_colloc=X_colloc, alpha=alpha, optimizer_type="both")

if flow_model == "dinucci":
    # dinucci flow model fitting K
    model = DiNucciNormalizedScaledPINNFlow(X_train, u_train, k, layers, 0, 1, scale_q=scale_q, 
            X_colloc=X_colloc, alpha=alpha, optimizer_type="both")

model.train(N_train)
u_pred, f_pred = model.predict(X_test) 

# Saving model predictions
groupname = "no_colloc"
grp = out_file.create_group(groupname)
grp.create_dataset('n_colloc', data=n_colloc)
grp.create_dataset('u_pred', data=u_pred) 
grp.create_dataset('f_pred', data=f_pred) 


######################################################################
# 2. Train model using no collocation points
n_colloc = 30
alpha_colloc = 10*alpha
X_colloc = make_collocation(q_colloc, n_colloc, 0, L, scale_q=scale_q)

if flow_model == "dupuit":
    # dinucci flow model fitting K
    model = DupuitNormalizedScaledPINNFlow(X_train, u_train, k, layers, 0, 1, scale_q=scale_q, 
            X_colloc=X_colloc, alpha=alpha, alpha_colloc=alpha_colloc, optimizer_type="both")

if flow_model == "dinucci":
    # dinucci flow model fitting K
    model = DiNucciNormalizedScaledPINNFlow(X_train, u_train, k, layers, 0, 1, scale_q=scale_q, 
            X_colloc=X_colloc, alpha=alpha, alpha_colloc=alpha_colloc, optimizer_type="both")

model.train(N_train)
u_pred, f_pred = model.predict(X_test) 

# Saving model predictions
groupname = "colloc"
grp = out_file.create_group(groupname)
grp.create_dataset('n_colloc', data=n_colloc)
grp.create_dataset('u_pred', data=u_pred) 
grp.create_dataset('f_pred', data=f_pred) 

out_file.close()

