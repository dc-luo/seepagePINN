import scipy.io 
import os 
import numpy as np 

def load_data(name, n, data_dir="data/steady", non_dim=True, scale_q=1.0):
    """ loads dataset n"""
    data = scipy.io.loadmat(data_dir + "/%s_exp%d.mat" %(name, n)) 
    Q = data['Q'][0][0]
    K_truth = data['K'][0][0]
    
    x_data = data['xexp'][:,0]
    u_data = data['hexp']
    # print(x_data)
    L = data['L'][0][0]
    W = data['W'][0][0]

    # x_data = L - x_data 
    q_data = -np.ones(x_data.shape) * Q/W  / scale_q

    if non_dim:
        x_data /= L
        u_data /= L
        q_data /= L*K_truth
    X_data = np.stack((x_data, q_data)).T

    return X_data, u_data, L, W, K_truth 

def load_all(name, n_max, non_dim=True, scale_q=1.0):
    """ load all training data into a dictionary 
    stored in order of X, u, L, W, k""" 
    training_data = dict() 
    for i in range(n_max):
        training_data[i+1] = load_data(name, i+1, non_dim, scale_q)

    return training_data 


def load_data_name(name, data_dir="data/steady", non_dim=False, subsample=200, scale_q=1.0):
    """ load dataset given full name of data file

    Notes: 
    x_data is of shape (1,n)
    u_data is of shape (n,1)
    L is measured in mm 
    Some entries of u_data are Nan. Remove these elements 
    """
    MM_TO_M = 1000

    data = scipy.io.loadmat(data_dir + "/" + name)
    Q = data['Q'][0][0]
    K_truth = data['K'][0][0]
    
    x_data = data['xexp'][0,:]
    u_data = data['hexp']

    # Removing the Nan entries 
    x_data = x_data[~np.isnan(u_data[:,0])]
    u_data = u_data[~np.isnan(u_data)]

    if subsample < x_data.shape[0]:
        inds = np.sort(np.random.choice(x_data.shape[0], size=subsample, replace=False))
        x_data = x_data[inds]
        u_data = u_data[inds]

    u_data = u_data.reshape((-1, 1))
    L = data['L'][0][0]/MM_TO_M
    W = data['W'][0][0]

    q_data = -np.ones(x_data.shape) * Q/W  / scale_q

    if non_dim:
        x_data /= L
        u_data /= L
        q_data /= L*K_truth

    X_data = np.stack((x_data, q_data)).T

    return X_data, u_data, L, W, K_truth 


def load_all_dir(data_dir="data/steady", non_dim=False, subsample=200, scale_q=1.0):
    """ loads data for each mat file in a given directory """ 
    all_files = os.listdir(data_dir) 
    training_data = dict() 

    training_data = [] 
    for i_file in all_files:
        if i_file.endswith(".mat"):
            data_file = load_data_name(i_file, data_dir, non_dim=non_dim, 
                    subsample=subsample, scale_q=scale_q)
            training_data.append(data_file)

    
    # for count, i_file in enumerate(all_files):
    #     if i_file.endswith(".mat"):
    #         training_data[count] = load_data_name(i_file, data_dir, non_dim=non_dim, 
    #                 subsample=subsample, scale_q=scale_q)

    return training_data


