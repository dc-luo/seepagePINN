import sys 
import time 
import logging
import os

import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
import h5py 
import argparse

sys.path.append("../")
from seepagePINN import *

FIGSIZE=(12,8)

def parse_args():
    # Parsing in PDE model as input 
    parser = argparse.ArgumentParser(description='Select PDE model') 
    parser.add_argument('-c', '--case', type=str, default='1mm', choices=["1mm", "2mm"], help="Case name")
    parser.add_argument('-n', '--N_epoch', type=int, default=20000, help="Number of training iterations with ADAM")
    parser.add_argument("-r", "--random", help="Do not set constant seed", action="store_true", default=False)
    parser.add_argument("-i", "--index", type=int, default=0, help="Index of the data set")
    parser.add_argument("--regularization", type=str, default="average", choices=["average", "max"], help="selection of regularization parameter")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if not args.random:
        # Set random seed
        np.random.seed(SEEPAGE_FIXED_SEED)
        tf.set_random_seed(SEEPAGE_FIXED_SEED) 
    
    ######################################################################
    case = args.case
    data_dir = "data/" + case
    scale_q = 1e-4
    subsample = 200
    training_data = load_all_dir(data_dir, scale_q=scale_q, subsample=subsample) 
    
    training_list = [args.index]
    X_train, u_train = make_training_set(training_list, training_data) 
    X, u, L, W, K = training_data[args.index] 
    if case == "1mm":
        K = K/10 # incorrect recording from dataset
    
    X_colloc = None

    test_list = [args.index]
    X_test, u_test = make_training_set(test_list, training_data) 
    
    ######################################################################
    alpha = optimal_alpha(u_train, method=args.regularization)
    
    # Define models
    N_epoch = args.N_epoch
    layers = [2, 20, 20, 20, 20, 20, 1]
    
    model_dinucci = DiNucciNormalizedScaledPINNFitK(X_train, u_train, 2*K, layers, 0, 1, scale_q=scale_q, 
            X_colloc=X_colloc, alpha=alpha, optimizer_type="both")
    model_dinucci.train(N_epoch)
    
    ######################################################################
    # Post processing and save output as hdf5
    save_path = "experimental/terms/"
    os.makedirs(save_path, exist_ok=True)
    out_file = h5py.File(save_path + case + "%s"%(args.index) + ".h5", 'w')
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
    # Save dinucci result 
    u_pred, f_pred = model_dinucci.predict(X_test)
    K_dinucci = np.exp(model_dinucci.sess.run(model_dinucci.lambda_1)[0])
    
    # Saving model predictions
    groupname = "dinucci"
    grp = out_file.create_group(groupname)
    grp.create_dataset('alpha', data=alpha)
    grp.create_dataset('K', data=K_dinucci)
    grp.create_dataset('u_pred', data=u_pred) 
    grp.create_dataset('f_pred', data=f_pred) 
    
    f1_pred, f2_pred, f3_pred = model_dinucci.predict_terms(X_test)
    grp.create_dataset('f1_pred', data=f1_pred) 
    grp.create_dataset('f2_pred', data=f2_pred) 
    grp.create_dataset('f3_pred', data=f3_pred) 
    
    out_file.close()
    ######################################################################

if __name__ == "__main__":
    plt.rcParams.update({'font.size' : 16})
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    main()
