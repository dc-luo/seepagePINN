import matplotlib.pyplot as plt
import numpy as np 

SEEPAGE_FIXED_SEED = 12345

def optimal_alpha(u_train, method="average"):
    assert method == "average" or method == "max"
    if method == "average":
        alpha = np.mean(u_train[:,0]**2)
    else:
        alpha = np.max(u_train[:,0])**2
    return alpha


def plot_data(X, u, max_L=None, max_h1=None):
    plt.plot(X[:,0], u[:,0], '-or') 
    plt.xlim(0, max_L)
    plt.ylim(0, max_h1)
    plt.grid(True)


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


