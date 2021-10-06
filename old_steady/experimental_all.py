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
from pinn_models import DupuitNormalizedScaledPINNFitK, DiNucciNormalizedScaledPINNFitK, DiNucciNormalizedScaledPINNFitAll
from utils import *

FIGSIZE=(12,8)
plt.rcParams.update({'font.size' : 16})
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

######################################################################
# Parsing in PDE model as input 
parser = argparse.ArgumentParser(description='Select PDE model') 
parser.add_argument('-c', '--case', type=str, default='1mm', help="case name")
parser.add_argument('-n', '--N_epoch', type=int, default=20000, help="Number of training epochs")
parser.add_argument("-r", "--random", help="Do not set constant seed", action="store_true", default=False)
args = parser.parse_args()

######################################################################
if not args.random:
    # Set random seed
    np.random.seed(1234)
    tf.set_random_seed(1234) 

######################################################################
case = args.case
data_dir = "data/" + case
scale_q = 1e-4
subsample=200
training_data = load_all_dir(data_dir, scale_q=scale_q, subsample=subsample) 

plt.figure()
plot_scatter_all(training_data, scale_q=scale_q)
plt.show()

training_list = [i for i in range(3, 10)]
X_train, u_train = make_training_set(training_list, training_data) 
X, u, L, W, K = training_data[1] 
if case == "1mm":
    K = K/10

X_colloc = None

test_list = [i for i in range(1, len(training_data)+1)]
X_test, u_test = make_training_set(test_list, training_data) 

######################################################################
h_max_train = np.max(u_train[:,0])
alpha_large = 1 * h_max_train**2

# Define models
N_epoch = args.N_epoch
layers = [2, 20, 20, 20, 20, 20, 1]


model_dupuit = DupuitNormalizedScaledPINNFitK(X_train, u_train, 2*K, layers, 0, L, scale_q=scale_q, 
        X_colloc=X_colloc, alpha=alpha_large, optimizer_type="both")
model_dupuit.train(N_epoch)

model_dinucci = DiNucciNormalizedScaledPINNFitK(X_train, u_train, 2*K, layers, 0, L, scale_q=scale_q, 
        X_colloc=X_colloc, alpha=alpha_large, optimizer_type="both")
model_dinucci.train(N_epoch)

model_none = DiNucciNormalizedScaledPINNFitK(X_train, u_train, 2*K, layers, 0, L, scale_q=scale_q, 
        X_colloc=X_colloc, alpha=0.0, optimizer_type="both")
model_none.train(N_epoch)


######################################################################
# Post processing and save output as hdf5
save_path = "paper/experimental/all/"
out_file = h5py.File(save_path + case + ".h5", 'w')
out_file.create_dataset('training_list', data=training_list)
out_file.create_dataset('n_total', data=len(training_data))
out_file.create_dataset('X_data', data=X_train)
out_file.create_dataset('u_data', data=u_train)
out_file.create_dataset('X_test', data=X_test)
out_file.create_dataset('u_test', data=u_test)
out_file.create_dataset('scale_q', data=scale_q)
out_file.create_dataset('K_truth', data=K)
out_file.create_dataset('L', data=L)


######################################################################
# Save Dupuit result 
u_pred, f_pred = model_dupuit.predict(X_test) 
K_dupuit = np.exp(model_dupuit.sess.run(model_dupuit.lambda_1)[0])

# Saving model predictions
groupname = "dupuit"
grp = out_file.create_group(groupname)
grp.create_dataset('alpha', data=alpha_large)
grp.create_dataset('K', data=K_dupuit)
grp.create_dataset('u_pred', data=u_pred) 
grp.create_dataset('f_pred', data=f_pred) 

######################################################################
# Save dinucci result 
u_pred, f_pred = model_dinucci.predict(X_test)
K_dinucci = np.exp(model_dinucci.sess.run(model_dinucci.lambda_1)[0])

# Saving model predictions
groupname = "dinucci"
grp = out_file.create_group(groupname)
grp.create_dataset('alpha', data=alpha_large)
grp.create_dataset('K', data=K_dinucci)
grp.create_dataset('u_pred', data=u_pred) 
grp.create_dataset('f_pred', data=f_pred) 

f1_pred, f2_pred, f3_pred = model_dinucci.predict_terms(X_test)
grp.create_dataset('f1_pred', data=f1_pred) 
grp.create_dataset('f2_pred', data=f2_pred) 
grp.create_dataset('f3_pred', data=f3_pred) 

######################################################################
# Save blank result 
u_pred, f_pred = model_none.predict(X_test) 

# Saving model predictions
groupname = "none"
grp = out_file.create_group(groupname)
grp.create_dataset('alpha', data=0.0)
grp.create_dataset('u_pred', data=u_pred) 
grp.create_dataset('f_pred', data=f_pred) 

out_file.close()
######################################################################

# plot_scatter = True

# if plot_scatter:
#     # plot the 3d scatter for data     
#     u_pred_test, _ = model.predict(X_test)
#     plt.figure(figsize=FIGSIZE)
#     ax = plt.axes(projection='3d')
#     ax.view_init(22, 45)
#     ax.scatter3D(X_test[:,0], X_test[:,1], u_test[:,0], color='r')
#     ax.scatter3D(X_test[:,0], X_test[:,1], u_pred_test[:,0], color='b')
#     plt.xlabel("x")
#     plt.ylabel("q")
#     plt.title("large")
# 
#     u_pred_test, _ = model_small.predict(X_test)
#     plt.figure(figsize=FIGSIZE)
#     ax = plt.axes(projection='3d')
#     ax.view_init(22, 45)
#     ax.scatter3D(X_test[:,0], X_test[:,1], u_test[:,0], color='r')
#     ax.scatter3D(X_test[:,0], X_test[:,1], u_pred_test[:,0], color='b')
#     plt.xlabel("x")
#     plt.ylabel("q")
#     plt.title("small")
# 
#     plt.show()
