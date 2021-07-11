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

from pinn import DupuitPINN
from dinuccipinn import DinucciPINN
from fun import *

equation = False

load_model = False
savename = "unsteady_bc_noiseless"

if equation == True :

    # NN architecture
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    # PDE problem set up
    n = 50
    t_num = 10000
    t_num_boundary = 100
    
    T_total = 4.0
    L = 1.0
    x = np.linspace(0.0, L, n)
    t = np.linspace(0.0, T_total, t_num)

    dt = t[1] - t[0]
    dx = x[1] - x[0]
    H = 0.1
    a = 0.1
    b = 0
    K = 1.0
    
    alpha = 1
    betas = [1, 1]
    
    u_initial = H * np.ones(len(x))
    u0 = u_initial.copy()
    # u_initial[0] = 0
    
    FD_soloution = FD1D_seepage(u_initial,x,t,dx,dt,K,a)


    X, T = np.meshgrid(x,t)
    
    # X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    X_star = np.stack((X.flatten(), T.flatten())).T
    u_star = FD_soloution.flatten()[:,None]

    lb = X_star.min(0)
    ub = X_star.max(0)
    
    # points for boundary residual evaluation
    t_boundary = np.linspace(0, T_total, t_num_boundary)
    # t_boundary = t
    x_left_boundary = np.zeros(t_boundary.shape)
    x_right_boundary = L * np.ones(t_boundary.shape)
    X_left_boundary = np.stack((x_left_boundary, t_boundary)).T
    X_right_boundary = np.stack((x_right_boundary, t_boundary)).T
    

    add_noise = False
    noise_sd = 0.05 # as a percentage
    N_u = 2000

    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx,:]
    # X_u_train = X_star
    u_train = u_star[idx,:]
    # u_train = u_star



    if add_noise:
        u_train = u_train + noise_sd*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])

    if load_model:
        tf.reset_default_graph()
        model = DupuitPINN(X_u_train, X_left_boundary, X_right_boundary,
                        u_train, 1.0, 0.0, layers, lb, ub, b=b, alpha=alpha, betas=betas, optimizer_type="adam")
        model.load(savename)

        # u_pred, f_pred = model.predict(X_star)
        u_pred, f_pred, f_left, f_right = model.predict(X_u_train)
        k_pred = np.exp(model.sess.run(model.lambda_1))

        FD_sol_fit = FD1D_seepage(u0,x,t,dx,dt,k_pred,a)

        print("k truth: ", K)
        print("k recovered: ", k_pred)
        print("relative error: ", np.abs(K-k_pred)/np.abs(K))

        ind_tests = np.sort(np.random.choice(np.arange(0,t_num), 10, replace=False))
        ind_tests = np.arange(0,t_num, int(t_num/50))

        # plot_animation(x, dt, FD_soloution, model, ind_tests, FD_sol_fit)
        print("k: ", model.sess.run(model.lambda_1))
        print("a: ", model.sess.run(model.lambda_2))

        #### Generate plot of the data ####

        plt.rcParams.update({"font.size" : 16})
        plt.figure(figsize=(8,4))
        p = plt.pcolor(T, X, FD_soloution, cmap="jet")
        plt.colorbar(p)
        plt.xlabel("t")
        plt.ylabel("x")
        plt.plot(X_u_train[:,1], X_u_train[:,0], 'xk')
        plt.savefig("unsteady_training_data.png")
        plt.close()


        ind_tests = np.arange(0,t_num, int(t_num/3))
        ind_tests = np.append(ind_tests, t_num-1)
        plot_timestamps(x, dt, FD_soloution, model, ind_tests, FD_sol_fit)

    else:
    # Train model
        model = DupuitPINN(X_u_train, X_left_boundary, X_right_boundary,
                        u_train, 1.0, 0.0, layers, lb, ub, b=b, alpha=alpha, betas=betas, optimizer_type="adam")
        model.train(5000)

    # u_pred, f_pred = model.predict(X_star)
        u_pred, f_pred, f_left, f_right = model.predict(X_u_train)
        k_pred = np.exp(model.sess.run(model.lambda_1))

        FD_sol_fit = FD1D_seepage(u0,x,t,dx,dt,k_pred,a)

        print("k truth: ", K)
        print("k recovered: ", k_pred)
        print("relative error: ", np.abs(K-k_pred)/np.abs(K))

        ind_tests = np.sort(np.random.choice(np.arange(0,t_num), 10, replace=False))
        ind_tests = np.arange(0,t_num, int(t_num/50))

        plot_animation(x, dt, FD_soloution, model, ind_tests, FD_sol_fit)
        model.save(savename)

else:
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    
    h1 = 1.0
    H2 = 0.167
    L = 0.667
    K = 1.0
    eta = 0.4
    bounds = [0.5,1.0]
    T_max = 2
    N_steps = 4000
    N = 64
    Dinucci_h, Dinucci_q, x ,t  = Dinucci_seepage(h1,H2,L,K,eta,bounds,T_max,N_steps,N)
    
#    plot_fd_solution(x, Dinucci_h, i=1000, format_str="-")
#    plt.show()
#    exit(1)
    X, T = np.meshgrid(x[:,0],np.array(t))
    
    # X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    X_star = np.stack((X.flatten(), T.flatten())).T

    print(X_star)
    #X_star = np.concatenate([x,t],1)
     
    u_star  = Dinucci_h.flatten()[:,None]
    q_star  = Dinucci_q.flatten()[:,None]

    lb = X_star.min(0)
    ub = X_star.max(0)
    
    add_noise = False
    noise_sd = 0.05 # as a percentage
    N_u = 2000

    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_train = X_star[idx,:]
    u_train = u_star[idx,:]
    q_train = q_star[idx,:]

    if add_noise:
        u_train = u_train + noise_sd*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
        q_train = q_train + noise_sd*np.std(q_train)*np.random.randn(q_train.shape[0], q_train.shape[1])

    
    alpha =  1.0
    gamma =  1.0
    betas =  [1.0,1.0]
   
    qh = ((h1**2 - H2**2)*K)/(2*L)
    # points for boundary residual evaluation
    t_boundary = np.linspace(0, T_max, 100)
    # t_boundary = t
    q_left_boundary = qh* np.ones(t_boundary.shape)
    h_right_boundary = H2 * np.ones(t_boundary.shape)
    X_left_boundary = np.stack((q_left_boundary, t_boundary)).T
    X_right_boundary = np.stack((h_right_boundary, t_boundary)).T
    

    add_noise = False
    noise_sd = 0.05 # as a percentage
    N_u = 2000

    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx,:]
    # X_u_train = X_star
    u_train = u_star[idx,:]
    

    model = DinucciPINN(X_train, X_left_boundary, X_right_boundary,
                        u_train, q_train, 0.5, layers, lb, ub, qh, H2, alpha=alpha, gamma=gamma,  betas=betas, optimizer_type="adam")
    model.train(5000)

    # u_pred, f_pred = model.predict(X_star)
    u_pred, q_pred, f1_pred, f2_pred, f_left, f_right = model.predict(X_train)
    eta_k_pred = np.exp(model.sess.run(model.lambda_1))
    #eta_pred = np.exp(model.sess.run(model.lambda_2))
