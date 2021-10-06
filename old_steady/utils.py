import scipy.io 
import os
import matplotlib.pyplot as plt
import numpy as np 


def plot_scatter_all(dataset, scale_q=1.0):
    """ 
    Plots an entire dataset as scatter plot 
    where X = [x, q]
    """    
    
    ax = plt.axes(projection='3d')
    ax.view_init(22, 45)
    for key in dataset.keys():
        X = dataset[key][0]
        u = dataset[key][1]
        ax.scatter3D(X[:,0], X[:,1]*scale_q, u[:,0], color='r')
    plt.xlabel("x")
    plt.ylabel("q")
    plt.title("Data")


def make_collocation(q_list, n_colloc, x0, x1, scale_q=1.0):
    """ Make training data for collocation 
    given q in q_list and x in linspace(x0, x1, n_colloc) 
    """
    assert(n_colloc >= 0)
    if n_colloc == 0:
        X_colloc = None
    else:
        x_colloc = np.array([]) 
        q_colloc = np.array([])
        for q in q_list: 
            x_locs = np.linspace(x0, x1, n_colloc)
            q_locs = q*np.ones(x_locs.shape) / scale_q 
            x_colloc = np.append(x_colloc, x_locs) 
            q_colloc = np.append(q_colloc, q_locs) 
        X_colloc = np.stack((x_colloc, q_colloc)).T 
    return X_colloc

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


def load_data_name(name, data_dir="data/steady", non_dim=False, subsample=200, scale_q=1.0):
    """ load dataset given full name 

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
    count = 1 
    for i_file in all_files:
        if i_file.endswith(".mat"):
            training_data[count] = load_data_name(i_file, data_dir, non_dim=non_dim, 
                    subsample=subsample, scale_q=scale_q)
            count += 1 

    return training_data

def load_all(name, n_max, non_dim=True, scale_q=1.0):
    """ load all training data into a dictionary 
    stored in order of X, u, L, W, k""" 
    training_data = dict() 
    for i in range(n_max):
        training_data[i+1] = load_data(name, i+1, non_dim, scale_q)

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


def plot_data(X, u, max_L=None, max_h1=None):
    plt.plot(X[:,0], u[:,0], '-or') 
    plt.xlim(0, max_L)
    plt.ylim(0, max_h1)
    plt.grid(True)


def dupuit_analytic(x, h2, q, L, K):
    return np.sqrt(2*q/K*(L-x)+h2**2)

def make_synthetic_set_dupuit(h2, q, L, K, n_points, scale_q=1.0, noise_sd=0):
    # Make head data
    W = 1
    x = np.linspace(0, L, n_points) 
    h = dupuit_analytic(x, h2, q, L, K) 
    h += np.random.randn(h.shape[0]) * noise_sd  
    
    qs = np.ones(x.shape) * q / scale_q 

    X_data = np.stack((x, qs)).T
    u_data = np.reshape(h, (h.shape[0], 1))

    return X_data, u_data, L, W, K

def make_synthetic_all_dupuit(q_list, h2, L, K, n_points, scale_q=1.0, noise_sd=0):
    training_data = dict() 
    for i, q in enumerate(q_list):
        training_data[i+1] = make_synthetic_set_dupuit(h2, q, L, K, n_points, scale_q=scale_q, noise_sd=noise_sd) 
    return training_data

def dinucci_analytic(x, h1, h2, L):
    q_K = (h1**2 - h2**2)/(2*L)
    h_eval = np.sqrt(- 2 * q_K * x 
            + 2/3 * (q_K)**2*(1 - np.exp(-3 * x /q_K))
            + h1**2)
    return h_eval

def make_synthetic_set_dinucci(h1, h2, L, K, n_points, scale_q=1.0, noise_sd=0):
    # Make head data
    W = 1
    x = np.linspace(0, L, n_points) 
    h = dinucci_analytic(x, h1, h2, L) 
    h += np.random.randn(h.shape[0]) * noise_sd  
    
    q_K = (h1**2 - h2**2)/(2*L)
    q = np.ones(x.shape) *q_K * K /scale_q

    X_data = np.stack((x, q)).T
    u_data = np.reshape(h, (h.shape[0], 1))

    return X_data, u_data, L, W, K

def make_synthetic_all_dinucci(h1_list, h2, L, K, n_points, scale_q=1.0, noise_sd=0):
    training_data = dict() 
    for i, h1 in enumerate(h1_list):
        training_data[i+1] = make_synthetic_set_dinucci(h1, h2, L, K, n_points, scale_q=scale_q, noise_sd=noise_sd) 
    return training_data

def make_synthetic_arrays_dinucci(h1_list, h2, L, K, n_points, scale_q=1.0, noise_sd=0):
    h1 = h1_list[0]
    X_train, u_train, L, W, K = make_synthetic_set_dinucci(h1, h2, L, K, n_points, scale_q=scale_q, noise_sd=noise_sd) 

    for h1 in h1_list[1:]:
        X, u, _, _, _ = make_synthetic_set_dinucci(h1, h2, L, K, n_points, scale_q=scale_q, noise_sd=noise_sd) 
        X_train = np.append(X_train, X, axis=0)
        u_train = np.append(u_train, u, axis=0)

    return X_train, u_train

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    h1 = 1
    h2 = 0.5
    L = 1.0
    K = 0.5
    noise_sd = 0.00*h1
    n_points = 50

    plt.figure()
    X, u, L, W, K = make_synthetic_set_dinucci(h1, h2, L, K, n_points, noise_sd=noise_sd) 
    plot_data(X, u)

    h2 = 0.0
    X, u, L, W, K = make_synthetic_set_dinucci(h1, h2, L, K, n_points, noise_sd=noise_sd) 
    plot_data(X, u)
    plt.show()


