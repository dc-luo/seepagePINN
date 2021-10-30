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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

FIGSIZE=(12,8)
plt.rcParams.update({'font.size' : 16})

def parse_args():
    ######################################################################
    # Parsing in PDE model as input 
    parser = argparse.ArgumentParser(description='Inputs for training') 
    parser.add_argument('-n', '--N_epoch', type=int, default=20000, help="Number of training epochs")
    parser.add_argument('-d', '--data_model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help="PDE choice for interpreting data: dinucci or dupuit")
    parser.add_argument("-r", "--random", help="Do not set constant seed", action="store_true", default=False)
    parser.add_argument("-p", "--positive", help="Set positive bounds on PDE coefficients", action="store_true", default=False)
    parser.add_argument('-K', '--K', type=float, default=1e-2, help="True hydraulic conductivity")
    parser.add_argument("--regularization", type=str, default="average", choices=["average", "max"], help="selection of regularization parameter")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if not args.random:
        # Set random seed
        np.random.seed(77777)
        tf.set_random_seed(77777)
    
    data_model = args.data_model
    
    print("Data generated using ", data_model)
    print("Random: ", args.random)
    print("K: ", args.K)
    
    ### Training data ###
    L = 1.0 
    K = args.K
    h2 = 0
    n_max = 1
    n_train = 1
    n_points = 100
    n_points_refined = 100
    
    noise_ratio = 0.01
    
    # Define training flow rates 
    q_min = 1e-3
    q_list = np.array([q_min])
    
    scale_q = q_min
    
    if data_model == 'dinucci':
        # Convert to h1
        h1_list = np.sqrt(2*L*q_list/K + h2**2)
        noise_sd = noise_ratio * np.max(h1_list)
        
        # Make synthetic dinucci data
        training_data = make_synthetic_all_dinucci(h1_list, h2, L, K, n_points, scale_q=scale_q, noise_sd=noise_sd)
        refined_data = make_synthetic_all_dinucci(h1_list, h2, L, K, n_points_refined, scale_q=scale_q, noise_sd=0)
    
    
    if data_model == 'dupuit':
        # First run with max q to determine an appropriate noise sd 
        h_max = dupuit_analytic(0, L, q_max, K, L)
        noise_sd = noise_ratio * h_max
    
        # Make synthetic dupuit data 
        training_data = make_synthetic_all_dupuit(q_list, h2, L, K, n_points, scale_q=scale_q, noise_sd=noise_sd)
        refined_data = make_synthetic_all_dupuit(q_list, h2, L, K, n_points_refined, scale_q=scale_q, noise_sd=0)
    
    
    if n_train < n_max:
        # Random selection of training data
        all_list = np.array([i for i in range(n_max)]) 
        training_list = np.random.choice(all_list, size=(n_train,), replace=False)
    else:
        # All data as training data
        training_list = [i for i in range(n_train)]
    
    X_train, u_train = make_training_set(training_list, training_data) 
    X, u, L, W, k = training_data[0] 
    
    # Test data - all as test data 
    test_list = [i for i in range(n_max)]
    X_test, u_test = make_training_set(test_list, refined_data)
    
    X_colloc = None
    
    ######################################################################
    if args.regularization == "max":
        alpha_reference = np.max(u_train[:,0])**2
    else:
        alpha_reference = np.mean(u_train[:,0]**2)
    
    print("Using ", args.regularization, " based reference regularization value: ", alpha_reference)
    alpha = alpha_reference
    
    # Define models
    N_epoch = args.N_epoch
    layers = [2, 20, 20, 20, 20, 20, 1]
    
    # Dinucci model 
    model_dinucci = DiNucciNormalizedScaledPINNFitK(X_train, u_train, k, layers, 0, 1, scale_q=scale_q, 
            X_colloc=X_colloc, alpha=alpha, optimizer_type="both")
    model_dinucci.train(N_epoch)
    
    ######################################################################
    # Post processing and save output as hdf5
    
    save_path = "synthetic/terms/"
    os.makedirs(save_path, exist_ok=True)
    save_name = save_path + "data_" + data_model
    save_name += "K%e" %(K)
    
    out_file = h5py.File(save_name + ".h5", "w")
    out_file.create_dataset('q', data=q_list)
    out_file.create_dataset('N_epoch', data=N_epoch)
    out_file.create_dataset('training_list', data=training_list)
    out_file.create_dataset('X_data', data=X_train)
    out_file.create_dataset('u_data', data=u_train)
    out_file.create_dataset('X_test', data=X_test)
    out_file.create_dataset('u_test', data=u_test)
    out_file.create_dataset('K_truth', data=K)
    out_file.create_dataset('L', data=L)
    out_file.create_dataset('noise_ratio', data=noise_ratio)
    
    # Dinucci predictions and saving 
    u_pred, f_pred = model_dinucci.predict(X_test) 
    f1_pred, f2_pred, f3_pred = model_dinucci.predict_terms(X_test)
    K_dinucci = np.exp(model_dinucci.sess.run(model_dinucci.lambda_1)[0])
    
    # Saving 
    groupname = "dinucci"
    grp = out_file.create_group(groupname)
    grp.create_dataset('alpha', data=alpha)
    grp.create_dataset('K', data=K_dinucci)
    grp.create_dataset('u_pred', data=u_pred) 
    grp.create_dataset('f_pred', data=f_pred) 
    grp.create_dataset('f1_pred', data=f1_pred) 
    grp.create_dataset('f2_pred', data=f2_pred) 
    grp.create_dataset('f3_pred', data=f3_pred) 
    
    out_file.close()
    
    print("#"*20)
    print("Summarizing run results")
    print("Data generated using ", data_model)
    print("Random: ", args.random)
    print("Dinucci: K = ", K_dinucci)
    
    plot_scatter = False
    
    if plot_scatter:
        # plot the 3d scatter for data     
        u_pred_test, _ = model_dinucci.predict(X_test)
        plt.figure(figsize=FIGSIZE)
        ax = plt.axes(projection='3d')
        ax.view_init(22, 45)
        ax.scatter3D(X_test[:,0], X_test[:,1], u_test[:,0], color='r')
        ax.scatter3D(X_test[:,0], X_test[:,1], u_pred_test[:,0], color='b')
        plt.xlabel("x")
        plt.ylabel("q")
        plt.title("DiNucci")
        plt.show()
    
if __name__ == "__main__":
    main()
