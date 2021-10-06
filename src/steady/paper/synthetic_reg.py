import sys 
import time 
import logging
import os
import argparse

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import h5py 

sys.path.append("../")
from seepagePINN import *

# Tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Matplotlib 
FIGSIZE=(12,8)
plt.rcParams.update({'font.size' : 16})

######################################################################
# Parsing in PDE model as input 
parser = argparse.ArgumentParser(description='Select PDE model') 
parser.add_argument('-n', '--N_epoch', type=int, default=20000, help="Number of training epochs")
parser.add_argument('-m', '--model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help="PDE choice: dinucci or dupuit")
parser.add_argument("-r", "--random", help="Do not set constant seed", action="store_true", default=False)
args = parser.parse_args()

######################################################################
if not args.random:
    # Set random seed
    np.random.seed(SEEPAGE_FIXED_SEED)
    tf.set_random_seed(SEEPAGE_FIXED_SEED) 

flow_model = args.model

print("Data generated using ", flow_model)
print("NN interpret using ", flow_model)
print("Random: ", args.random)

### Training data ###
L = 1.0 
K = 0.2
h2 = 0
n_max = 8
n_train = 8
n_points = 30

noise_ratio = 0.05

# Define training flow rates 
q_min = 1e-4
q_max = 1e-3
q_list = np.linspace(q_min, q_max, n_max)
scale_q = q_min

n_points_refined = 100

if flow_model == 'dinucci':
    # Convert to h1
    h1_list = np.sqrt(2*L*q_list/K + h2**2)
    noise_sd = noise_ratio * np.max(h1_list)
    
    # Make synthetic dinucci data
    training_data = make_synthetic_all_dinucci(h1_list, h2, L, K, n_points, scale_q=scale_q, noise_sd=noise_sd)

    # Refined grid points at same training q values for plotting exact solution and PINN prediction
    refined_data = make_synthetic_all_dinucci(h1_list, h2, L, K, n_points_refined, scale_q=scale_q, noise_sd=0)

if flow_model == 'dupuit':
    # First run with max q to determine an appropriate noise sd 
    h_max = dupuit_analytic(0, L, q_max, K, L)
    noise_sd = noise_ratio * h_max

    # Make synthetic dupuit data 
    training_data = make_synthetic_all_dupuit(q_list, h2, L, K, n_points, scale_q=scale_q, noise_sd=noise_sd)

    # Refined grid points at same training q values for plotting exact solution and PINN prediction
    refined_data = make_synthetic_all_dupuit(q_list, h2, L, K, n_points_refined, scale_q=scale_q, noise_sd=0)

# Training data
training_list = [i+1 for i in range(n_train)]
X_train, u_train = make_training_set(training_list, training_data) 
X, u, L, W, k = training_data[1] 

# Test data 
test_list = [i+1 for i in range(n_max)]
X_test, u_test = make_training_set(test_list, refined_data)
X_colloc = None

### Neural network ### 
h_train = u_train[:,0] 

alpha_average = np.mean(h_train**2)
alpha_max = np.max(h_train**2)

print("Alpha via averaging: ", alpha_average)
print("Alpha via max: ", alpha_max)

alpha_scaling = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100] 
alphas = [alpha_average * a for a in alpha_scaling]
print(alphas)

n_alpha = len(alphas)

# Define models
N_train = args.N_epoch
layers = [2, 20, 20, 20, 20, 20, 1]
# layers = [2, 20, 20, 20, 20, 20, 20, 20, 1]

# Train model and save solutions/data to hdf5 format

save_path = "synthetic/regularization/"
os.makedirs(save_path, exist_ok=True)
out_file = h5py.File(save_path + "data_" + flow_model + ".h5", 'w')
out_file.create_dataset('q', data=q_list)
out_file.create_dataset('n_alpha', data=n_alpha)
out_file.create_dataset('X_data', data=X_train)
out_file.create_dataset('u_data', data=u_train)
out_file.create_dataset('X_test', data=X_test)
out_file.create_dataset('u_test', data=u_test)
out_file.create_dataset('noise_ratio', data=noise_ratio)
out_file.create_dataset('alpha_scaling', data=alpha_scaling)
out_file.create_dataset('alpha_reference', data=alpha_average)

for i, alpha in enumerate(alphas):
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
    u_train, f_train = model.predict(X_train)

    # Saving model predictions
    groupname = "alpha_%s" %(i) 
    grp = out_file.create_group(groupname)
    grp.create_dataset('alpha', data=alpha)

    # For trends
    grp.create_dataset('u_pred', data=u_pred) 
    grp.create_dataset('f_pred', data=f_pred) 
    
    # For training data
    grp.create_dataset('u_train', data=u_train) 
    grp.create_dataset('f_train', data=f_train) 


out_file.close()

