import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
import matplotlib.gridspec as gridspec
import time
import logging
import os
import h5py
import argparse

from pinn import DupuitPINN
from dinuccipinn import DinucciPINN
from fun import *
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'font.family': "Serif"})

# Tensorflow logging
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#FIGSIZE=(12,8)


######################################################################
# Parsing in PDE model as input
parser = argparse.ArgumentParser(description='Select PDE model')
parser.add_argument('-m', '--model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help="PDE choice: dinucci or dupuit")
parser.add_argument("-r", "--random", help="Do not set constant seed", action="store_true", default=False)
args = parser.parse_args()

######################################################################
if not args.random:
    # Set random seed
    np.random.seed(1234)
    tf.set_random_seed(1234)
    
    
equation = args.model

load_model = False
savename = "unsteady_DiNucci"

if equation == "dupuit" :

    # NN architecture
    layers = [2, 20, 20, 20, 20, 20, 20, 1]
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
    
    FD_soloution = FD1D_seepage(H,u_initial,x,t,dx,dt,K,a)
#    print(FD_soloution)
#
#    #New Contour plot video
#    print('Saving animation')
#    fps = 100000 # frame per sec
#
#    [len_t,len_x] = np.shape(FD_soloution) # frame number of the animation from the saved file
#    def update_plot(frame_number, zarray, plot,t):
#        fig.clear()
#        plot[0] = plt.plot(x,zarray[frame_number,:],'k--',label=r'$\mathcal{H}$')
#        #plot[0] = plt.plot(Grid.xc,phi_w(zarray[:,frame_number],carray[:,frame_number]),'b--',label=r'$\phi_w$')
#        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#        plt.xlabel("x")
#        plt.ylabel("h")
#        plt.ylim([np.min(FD_soloution)-0.05,np.max(FD_soloution)+0.05])
#        plt.xlim([0, L])
#        plt.title("Dupuit-Boussinesq Model = %0.2f" % t[frame_number],loc = 'center', fontsize=18)
#        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
##        plt.legend(loc='upper right',borderaxespad=0.)
#    fig = plt.figure(figsize=(10,10) , dpi=100)
#    plot = [plt.plot(x,FD_soloution[0,:],'k--',label=r'$\mathcal{H}$')]
##    plot = plt.plot(Grid.xc,phi_w(H_sol[:,0],C_sol[:,0]),'b--',label=r'$\phi_w$')
#    plt.xlabel("x")
#    plt.ylabel("h")
#    plt.ylim([np.min(FD_soloution)-0.05,np.max(FD_soloution)+0.05])
##    plt.title("Dupuit-Boussinesq Model = %0.2f" % t[i],loc = 'center', fontsize=18)
##    plt.title(r"$\tau$= %0.2f" % t[i],loc = 'center', fontsize=18)
#    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#    plt.legend(loc='upper right',borderaxespad=0.)
#    ani = animation.FuncAnimation(fig, update_plot, len_t, fargs=(FD_soloution[::100,:], plot[:],t[::100]), interval=1/fps)
#    ani.save(f"Dupuit.mov", writer='ffmpeg', fps=30)
#
#    exit(1)

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
    

    add_noise = True
    noise_sd = 0.05 # as a percentage
    N_u = 10000

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
                        u_train, 0.5, 0.0, layers, lb, ub, b=b, alpha=alpha, betas=betas, optimizer_type="adam")
        model.load(savename)

        # u_pred, f_pred = model.predict(X_star)
        u_pred, f_pred, f_left, f_right = model.predict(X_u_train)
        k_pred = np.exp(model.sess.run(model.lambda_1))

        FD_sol_fit = FD1D_seepage(H,u0,x,t,dx,dt,k_pred,a)

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
                        u_train, 0.5, 0.0, layers, lb, ub, b=0.1, alpha=alpha, betas=betas, optimizer_type="adam")
        model.train(40000)

    # u_pred, f_pred = model.predict(X_star)
        u_pred, f_pred, f_left, f_right = model.predict(X_u_train)
        k_pred = np.exp(model.sess.run(model.lambda_1))

        FD_sol_fit = FD1D_seepage(H,u0,x,t,dx,dt,k_pred,a)

        print("k truth: ", K)
        print("k recovered: ", k_pred)
        print("relative error: ", np.abs(K-k_pred)/np.abs(K))

        ind_tests = np.sort(np.random.choice(np.arange(0,t_num), 10, replace=False))
        ind_tests = np.arange(0,t_num, int(t_num/50))
        
        NN_u = model_solution(x, t, model)
        #plot_animation(x, dt, FD_soloution, model, ind_tests, FD_sol_fit)
        model.save(savename)
        

    #New Contour plot video
        print('Saving animation')
        fps = 100000 # frame per sec

        [len_t,len_x] = np.shape(FD_soloution) # frame number of the animation from the saved file
        def update_plot(frame_number,zarray,carray,varray, plot,t):
            fig.clear()
            plot[0] = plt.plot(x,zarray[frame_number,:],'--k',label="numerical")
            plot[0] = plt.plot(x,carray[frame_number,:],'--b',label="numerical with K recovered")
            plot[0] = plt.plot(x,varray[frame_number,:],'--r',label="NN")
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.xlabel("x")
            plt.ylabel("h")
            plt.ylim([0-0.05,0.15+0.05])
            plt.xlim([0, L])
            plt.title("Dupuit-NN-noise = %0.2f" % t[frame_number],loc = 'center', fontsize=18)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.legend(loc='upper right',borderaxespad=0.)
        fig = plt.figure(figsize=(10,10) , dpi=100)
        plot = plt.plot(x,FD_soloution[0,:],'--k',label="numerical")
        plot = plt.plot(x,FD_sol_fit[0,:],'--b',label="numerical with K recovered")
        plot = plt.plot(x,NN_u[0,:],'--r',label="NN")
#    plot = plt.plot(Grid.xc,phi_w(H_sol[:,0],C_sol[:,0]),'b--',label=r'$\phi_w$')
        plt.xlabel("x")
        plt.ylabel("h")
        plt.ylim([0-0.05,0.15+0.05])
#    plt.title("Dupuit-Boussinesq Model = %0.2f" % t[i],loc = 'center', fontsize=18)
#    plt.title(r"$\tau$= %0.2f" % t[i],loc = 'center', fontsize=18)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.legend(loc='upper right',borderaxespad=0.)
        ani = animation.FuncAnimation(fig, update_plot, len_t, fargs=(FD_soloution[::100,:],FD_sol_fit[::100,:],NN_u[::100,:], plot[:],t[::100]), interval=1/fps)
        ani.save(f"Dupuit-NN-noise.mov", writer='ffmpeg', fps=30)

if equation == "dinucci" :
    layers = [2, 20, 20, 20, 20, 20, 20, 2]
    
    h1 = 1
    H2 = 0.0167
    L = 0.667
    K = 1.0
    eta = 0.4
    bounds = [0.5,1.0]
    T_max = 2
    N_steps = 10000
    N = 106
    eta_k = eta/K
    Dinucci_h, Dinucci_q, x ,t  = Dinucci_seepage(h1,H2,L,eta_k,bounds,T_max,N_steps,N)
 
#     #New Contour plot video
#    print('Saving animation')
#    fps = 100000 # frame per sec
#
#    [len_t,len_x] = np.shape(Dinucci_h) # frame number of the animation from the saved file
#    def update_plot(frame_number, zarray, plot,t):
#        fig.clear()
#        plot[0] = plt.plot(x,zarray[frame_number,:],'k--',label=r'$\mathcal{H}$')
#        #plot[0] = plt.plot(Grid.xc,phi_w(zarray[:,frame_number],carray[:,frame_number]),'b--',label=r'$\phi_w$')
#        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#        plt.xlabel("x")
#        plt.ylabel("h")
#        plt.ylim([np.min(Dinucci_h)-0.05,np.max(Dinucci_h)+0.05])
#        plt.xlim([0, L])
#        plt.title("Di Nucci Model = %0.2f" % t[frame_number], loc = 'center', fontsize=18)
#        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
##        plt.legend(loc='upper right',borderaxespad=0.)
#    fig = plt.figure(figsize=(10,10) , dpi=100)
#    plot = [plt.plot(x,Dinucci_h[0,:],'k--',label=r'$\mathcal{H}$')]
##    plot = plt.plot(Grid.xc,phi_w(H_sol[:,0],C_sol[:,0]),'b--',label=r'$\phi_w$')
#    plt.xlabel("x")
#    plt.ylabel("h")
#    plt.ylim([np.min(Dinucci_h)-0.05,np.max(Dinucci_h)+0.05])
#    plt.title("Di Nucci Model",loc = 'center', fontsize=18)
##    plt.title(r"$\tau$= %0.2f" % t[i],loc = 'center', fontsize=18)
#    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#    plt.legend(loc='upper right',borderaxespad=0.)
#    ani = animation.FuncAnimation(fig, update_plot, len_t, fargs=(Dinucci_h[::100,:], plot[:],t[::100]), interval=1/fps)
#    ani.save(f"DiNucci.mov", writer='ffmpeg', fps=30)

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
    noise_sd = 0.1 # as a percentage
    N_u = 30000

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
    
    
    model = DinucciPINN(X_train, X_left_boundary, X_right_boundary,
                        u_train, q_train, 0.5, layers, lb, ub, qh, H2, alpha=alpha, gamma=gamma,  betas=betas, optimizer_type="adam")
    model.train(40000)

    # u_pred, f_pred = model.predict(X_star)
    u_pred, q_pred, f1_pred, f2_pred, f_left, f_right = model.predict(X_train)

    eta_k_pred = np.exp(model.sess.run(model.lambda_1))
    #eta_pred = np.exp(model.sess.run(model.lambda_2))

    Dinucci_h_fit, Dinucci_q_fit, x ,t  = Dinucci_seepage(h1,H2,L,eta_k_pred[0],bounds,T_max,N_steps,N)

    print("eta-k truth: ", eta_k)
    print("k recovered: ", eta_k_pred[0])
    print("relative error: ", np.abs(eta_k-eta_k_pred[0])/np.abs(eta_k))

    ind_tests = np.sort(np.random.choice(np.arange(0,N_steps), 10, replace=False))
    ind_tests = np.arange(0,N_steps, int(N_steps/50))

    plot_animation_di(x[:,0], T_max/N_steps, Dinucci_h, model, ind_tests, Dinucci_h_fit)
     
    
    model.save(savename)


#    print('Saving animation')
#    fps = 100000 # frame per sec
#    NN_u = model_solution_di(x[:,0], t, model)
#    [len_t,len_x] = np.shape(Dinucci_h) # frame number of the animation from the saved file
#    def update_plot(frame_number,zarray,carray,varray, plot,t):
#        fig.clear()
#        plot[0] = plt.plot(x,zarray[frame_number,:],'--k',label="numerical")
#        plot[0] = plt.plot(x,carray[frame_number,:],'--b',label="numerical with K recovered")
#        plot[0] = plt.plot(x,varray[frame_number,:],'--r',label="NN")
#        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#        plt.xlabel("x")
#        plt.ylabel("h")
#        plt.ylim([0-0.05,1+0.1])
#        plt.xlim([0, L])
#        plt.title("DiNucci-NN-10noise = %0.2f" % t[frame_number],loc = 'center', fontsize=18)
#        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#        plt.legend(loc='upper right',borderaxespad=0.)
#    fig = plt.figure(figsize=(10,10) , dpi=100)
#    plot = plt.plot(x,Dinucci_h[0,:],'--k',label="numerical")
#    plot = plt.plot(x,Dinucci_h_fit[0,:],'--b',label="numerical with K recovered")
#    plot = plt.plot(x,NN_u[0,:],'--r',label="NN")
##    plot = plt.plot(Grid.xc,phi_w(H_sol[:,0],C_sol[:,0]),'b--',label=r'$\phi_w$')
#    plt.xlabel("x")
#    plt.ylabel("h")
#    plt.ylim([0-0.05,0.15+0.05])
##    plt.title("Dupuit-Boussinesq Model = %0.2f" % t[i],loc = 'center', fontsize=18)
##    plt.title(r"$\tau$= %0.2f" % t[i],loc = 'center', fontsize=18)
#    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#    plt.legend(loc='upper right',borderaxespad=0.)
#    ani = animation.FuncAnimation(fig, update_plot, len_t, fargs=( Dinucci_h[::100,:], Dinucci_h_fit[::100,:],NN_u[::100,:], plot[:],t[::100]), interval=1/fps)
#    ani.save(f"DiNucci-NN-10noise.mov", writer='ffmpeg', fps=30)
    ni = pd.read_csv('model.csv')
    names = ['x','h']
    xni = ni['x'].map(lambda x: float(x))
    hni = ni['h'].map(lambda x: float(x))

    h0 = hni[0:106]
    x0 = xni[0:106]
    h1 = hni[106:212]
    x1 = xni[106:212]
    h2 = hni[212:318]
    x2 = xni[212:318]
    h3 = hni[318:424]
    x3 = xni[318:424]
    h4 = hni[424:530]
    x4 = xni[424:530]
    h5 = hni[530:636]
    x5 = xni[530:636]
    h6 = hni[636:742]
    x6 = xni[636:742]
    
    xx = x[:,0]
    in_list = [0,250,500,750,1000,1250,10000]
    
    tt0 = np.ones(xx.shape) * 0
    X0 = np.stack((xx, tt0)).T
    u_pred0, _, _, _, _, _ = model.predict(X0)

    tt1 = np.ones(xx.shape) * 0.05
    X1 = np.stack((xx, tt1)).T
    u_pred1, _, _, _, _, _ = model.predict(X1)
    
    tt2 = np.ones(xx.shape) * 0.1
    X2 = np.stack((xx, tt2)).T
    u_pred2, _, _, _, _, _ = model.predict(X2)
    
    tt3 = np.ones(xx.shape) * 0.15
    X3 = np.stack((xx, tt3)).T
    u_pred3, _, _, _, _, _ = model.predict(X3)
    
    tt4 = np.ones(xx.shape) * 0.2
    X4 = np.stack((xx, tt4)).T
    u_pred4, _, _, _, _, _ = model.predict(X4)
    
    tt5 = np.ones(xx.shape) * 0.25
    X5 = np.stack((xx, tt5)).T
    u_pred5, _, _, _, _, _ = model.predict(X5)
    
    tt6 = np.ones(xx.shape) * 2
    X6 = np.stack((xx, tt6)).T
    u_pred6, _, _, _, _, _ = model.predict(X6)

    
    
    plt.figure(figsize=(8,5))
    plt.plot(x0,h0,'+r', label="model")
    plt.plot(x1,h1,'+r')
    plt.plot(x2,h2,'+r')
    plt.plot(x3,h3,'+r')
    plt.plot(x4,h4,'+r')
    plt.plot(x5,h5,'+r')
    plt.plot(x6,h6,'+r')
    plt.plot(xx, Dinucci_h[0,:],'-k', label="numerical (K=0.4)")
    plt.plot(xx, Dinucci_h[250,:],'-k')
    plt.plot(xx, Dinucci_h[500,:],'-k')
    plt.plot(xx, Dinucci_h[750,:],'-k')
    plt.plot(xx, Dinucci_h[1000,:],'-k')
    plt.plot(xx, Dinucci_h[1250,:],'-k')
    plt.plot(xx, Dinucci_h[10000,:],'-k')
    plt.plot(xx, Dinucci_h_fit[0,:],'-b', label="numerical with K recovered (0.397)")
    plt.plot(xx, Dinucci_h_fit[250,:],'-b')
    plt.plot(xx, Dinucci_h_fit[500,:],'-b')
    plt.plot(xx, Dinucci_h_fit[750,:],'-b')
    plt.plot(xx, Dinucci_h_fit[1000,:],'-b')
    plt.plot(xx, Dinucci_h_fit[1250,:],'-b')
    plt.plot(xx, Dinucci_h_fit[10000,:],'-b')
    plt.plot(xx, u_pred0, '-y',label="Neural Networks (PINNs)")
    plt.plot(xx, u_pred1, '-y')
    plt.plot(xx, u_pred2, '-y')
    plt.plot(xx, u_pred3, '-y')
    plt.plot(xx, u_pred4, '-y')
    plt.plot(xx, u_pred5, '-y')
    plt.plot(xx, u_pred6, '-y')
    plt.xlabel("x")
    plt.ylabel("h")
    plt.title("Free surface elevations")
    plt.legend(loc='upper right',borderaxespad=0.)
    plt.show()
