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


def argument_parsing():
    """ Parsing in PDE model as input """
    parser = argparse.ArgumentParser(description='Select PDE model') 
    parser.add_argument('-c', '--case', type=str, default='1mm', choices=["1mm", "2mm"], help="Case name")
    parser.add_argument('-n', '--N_epoch', type=int, default=20000, help="Number of training epochs")
    parser.add_argument('-N', '--N_training', type=int, default=6, help="Number of training sets")
    parser.add_argument("-r", "--random", help="Do not set constant seed", action="store_true", default=False)
    parser.add_argument("--regularization", type=str, default="average", choices=["average", "max"], help="selection of regularization parameter")
    args = parser.parse_args()
    return args


def run_model(model_name, save_path, args, X_train, u_train, X_test, K, layers, L, scale_q, alpha):
    """ Make, train, then save model outputs as hdf5 """ 
    is_random = args.random
    N_epoch = args.N_epoch
    print("Running for ", model_name)
    print("Alpha = ", alpha)

    PERTURB_K_SCALING = 3

    if model_name == "dupuit_fit":
        model = DupuitNormalizedScaledPINNFitK(X_train, u_train, PERTURB_K_SCALING*K, layers, 0, 1, 
            scale_q=scale_q, X_colloc=None, alpha=alpha, optimizer_type="both")
        model.train(N_epoch)

    elif model_name == "dinucci_fit":
        model = DiNucciNormalizedScaledPINNFitK(X_train, u_train, PERTURB_K_SCALING*K, layers, 0, 1, 
            scale_q=scale_q, X_colloc=None, alpha=alpha, optimizer_type="both")
        model.train(N_epoch)

    elif model_name == "dupuit_flow":
        model = DupuitNormalizedScaledPINNFlow(X_train, u_train, K, layers, 0, 1, 
            scale_q=scale_q, X_colloc=None, alpha=alpha, optimizer_type="both")

    elif model_name == "dinucci_flow":
        model = DiNucciNormalizedScaledPINNFlow(X_train, u_train, K, layers, 0, 1, 
            scale_q=scale_q, X_colloc=None, alpha=alpha, optimizer_type="both")

    elif model_name == "vanilla":
        model = DiNucciNormalizedScaledPINNFlow(X_train, u_train, K, layers, 0, 1, 
            scale_q=scale_q, X_colloc=None, alpha=0, optimizer_type="both")

    else:
        print("Incorrect model input") 
        return 1

    # Training the Model
    if model_name == "vanilla":
        model.train(N_epoch)
    else:
        model.train(N_epoch)

    # Save outputs
    u_pred, f_pred = model.predict(X_test)
    out_file = h5py.File(save_path + model_name + ".h5", 'w')
    out_file.create_dataset("u_pred", data=u_pred)
    out_file.create_dataset("f_pred", data=f_pred)
    if model_name == "dupuit_fit" or model_name == "dinucci_fit":
        K = np.exp(model.sess.run(model.lambda_1)[0])
        out_file.create_dataset("K", data=K)

    out_file.close()
    tf.keras.backend.clear_session()


def main():
    """ Main routine for running multiple models """
    # Initialize 
    FIGSIZE=(12,8)
    plt.rcParams.update({'font.size' : 16})
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)

    # Parse argments
    args = argument_parsing()

    if not args.random:
        # Set random seed
        np.random.seed(77777)
        tf.set_random_seed(77777) 

    # Load data
    case = args.case
    data_dir = "data/" + case
    scale_q = 1e-4
    subsample = 200
    all_data = load_all_dir(data_dir, scale_q=scale_q, subsample=subsample) 


    # Form training set
    # training_list = [i for i in range(3, 10)]
    N_data = len(all_data)
    N_training = np.min([args.N_training, N_data])
    training_list = np.random.choice(N_data, N_training, replace=False)


    print("Running experimental data for ", args.case)
    print("Total number of datasets: ", N_data)
    print("Training sets: ", training_list)
    print(all_data)

    X_train, u_train = make_training_set(training_list, all_data) 
    X, u, L, W, K = all_data[0] 

    if case == "1mm":
        # Fixing incorrect data entry from 1mm data
        K = K/10
    
    # Form test set
    test_list = [i for i in range(N_data)]
    X_test, u_test = make_training_set(test_list, all_data) 
    print("Test sets: ", test_list)
    print("Measured K: ", K)
    print("Scale of q: ", scale_q)

    
    ######################################################################
    alpha = optimal_alpha(u_train, method=args.regularization)
    # Define models
    N_epoch = args.N_epoch
    layers = [2, 20, 20, 20, 20, 20, 1]
    
    # Post processing and save output as hdf5
    save_path = "experimental/all_models/%s/" %(args.case)
    os.makedirs(save_path, exist_ok=True)
    out_file = h5py.File(save_path + "data.h5", 'w')
    out_file.create_dataset('training_list', data=training_list)
    out_file.create_dataset('layers', data=layers)
    out_file.create_dataset('N_sets', data=len(all_data))
    out_file.create_dataset('X_data', data=X_train)
    out_file.create_dataset('u_data', data=u_train)
    out_file.create_dataset('X_test', data=X_test)
    out_file.create_dataset('u_test', data=u_test)
    out_file.create_dataset('N_epoch', data=N_epoch)
    
    out_file.create_dataset('K_truth', data=K)
    out_file.create_dataset('L', data=L)
    out_file.create_dataset('scale_q', data=scale_q)
    out_file.create_dataset('alpha', data=alpha)

    model_names = ["dupuit_fit", "dupuit_flow", "dinucci_fit", "dinucci_flow", "vanilla"]
    for model_name in model_names:
        run_model(model_name, save_path, args, X_train, u_train, X_test, K, layers, L, scale_q, alpha)
    
if __name__ == "__main__":
    main()
    
#     os.makedirs(save_path + "dupuit_fit", exist_ok=True)
#     os.makedirs(save_path + "dinucci_fit", exist_ok=True)
#     os.makedirs(save_path + "dupuit_flow", exist_ok=True)
#     os.makedirs(save_path + "dinucci_flow", exist_ok=True)
#     os.makedirs(save_path + "vanilla", exist_ok=True)
#     
#     
#     # Train all the neural networks
#     dupuit_fit = DupuitNormalizedScaledPINNFitK(X_train, u_train, K, layers, 0, L, scale_q=scale_q, 
#         X_colloc=X_colloc, alpha=alpha, optimizer_type="both")
#     dupuit_fit.train(N_epoch)
#     # dupuit_fit.save(save_path + "dupuit_fit/model")
#     # tf.keras.backend.clear_session()
#     
#     dinucci_fit = DiNucciNormalizedScaledPINNFitK(X_train, u_train, K, layers, 0, L, scale_q=scale_q, 
#         X_colloc=X_colloc, alpha=alpha, optimizer_type="both")
#     dinucci_fit.train(N_epoch)
#     # dinucci_fit.save(save_path + "dinucci_fit/model")
#     # tf.keras.backend.clear_session()
#         
#     dupuit_flow = DupuitNormalizedScaledPINNFlow(X_train, u_train, K, layers, 0, 1, scale_q=scale_q, 
#         X_colloc=X_colloc, alpha=alpha, optimizer_type="both")
#     dupuit_flow.train(N_epoch)
#     # dupuit_flow.save(save_path + "dupuit_flow/model")
#     # tf.keras.backend.clear_session()
#         
#     dinucci_flow = DiNucciNormalizedScaledPINNFlow(X_train, u_train, K, layers, 0, 1, scale_q=scale_q, 
#         X_colloc=X_colloc, alpha=alpha, optimizer_type="both")
#     dinucci_flow.train(N_epoch)
#     # dinucci_flow.save(save_path + "dinucci_flow/model")
#     # tf.keras.backend.clear_session()
#     
#     vanilla_nn = DiNucciNormalizedScaledPINNFlow(X_train, u_train, K, layers, 0, 1, scale_q=scale_q, 
#         X_colloc=X_colloc, alpha=0, optimizer_type="both")
#     vanilla_nn.train(N_epoch)
#     # dinucci_flow.save(save_path + "vanilla/model")
#     
#     groupnames = ["dupuit_fit", "dinucci_fit", "dupuit_flow", "dinucci_flow", "vanilla"]
#     models = [dupuit_fit, dinucci_fit, dupuit_flow, dinucci_flow, vanilla_nn]
#     
#     for groupname, model in zip(groupnames, models):
#         u_pred, f_pred = model.predict(X_test)
#         print(groupname)
#         grp = out_file.create_group(groupname)
#         grp.create_dataset("u_pred", data=u_pred)
#         grp.create_dataset("f_pred", data=f_pred)
#         if groupname == "dupuit_fit" or groupname == "dinucci_fit":
#             K = np.exp(model.sess.run(model.lambda_1)[0])
#             grp.create_dataset("K", data=K)
#     
#     out_file.close()
