import numpy as np 

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
