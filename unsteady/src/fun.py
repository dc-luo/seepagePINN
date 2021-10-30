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
import dolfin as dl
from matplotlib.animation import FuncAnimation
dl.set_log_active(False)

from pinn import DupuitPINN

# Tensorflow logging
os.environ['KMP_DUPLICATE_LIB_OK']='True'
FIGSIZE=(12,8)
np.random.seed(1234)
tf.set_random_seed(1234)

def make_collocation2D(t_time, n_colloc, x0, x1, scale_t=1.0):
    """ Make training data for collocation
    given q in q_list and x in linspace(x0, x1, n_colloc)
    """
    assert(n_colloc >= 0)
    if n_colloc == 0:
        X_colloc = None
    else:
        x_colloc = np.array([])
        t_colloc = np.array([])
        for t in t_time:
            x_locs = np.linspace(x0, x1, n_colloc)
            t_locs = t*np.ones(x_locs.shape) / scale_t
            x_colloc = np.append(x_colloc, x_locs)
            t_colloc = np.append(t_colloc, t_locs)
        X_colloc = np.stack((x_colloc, t_colloc)).T
    return X_colloc
    
def FD1D_seepage(h1,u_initial,x,t,dx,dt,K,a):
    u_old = u_initial
    u_new = u_initial
    u = []
    u.extend(u_new)
    const = K * dt / (dx * dx)
    for itime in range (1,len(t)):
        for idx in range(1,len(x)-1):
           u_new[idx] = const * ((u_old[idx+1] - u_old[idx])**2 + u_old[idx]*(u_old[idx+1] - 2*u_old[idx] + u_old[idx-1]))+ u_old[idx]
        
        #if boundary on the left hand size
        u_new[0] = u_new[1] - a * dx

        #if boundary on the right hand size
        u_new[len(x)-1] = h1
        
        #update u
        u.extend(u_new)
        u_old = u_new
    u = np.reshape(u,(len(t),len(x)))
    return u

def compute_first_derivative(func, x, delta):
    f1 = func(x)
    f2 = func(x+delta)
    return (f2-f1)/delta

def compute_second_derivative(func, x, delta):
    f0 = func(x-delta)
    f1 = func(x)
    f2 = func(x+delta)
    return (f2 - 2*f1 + f0)/delta**2

class InitialConditions(dl.UserExpression):
    def __init__(self, h1, **kwargs):
        self.h1 = h1
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = self.h1
        values[1] = 0.0

    def value_shape(self):
        return (2,)


def Dinucci_seepage(h1,H2,L,eta_k,bounds,T_max,N_steps,N):
    # Compute flow boundary condition
    q_star = -(h1**2 - H2**2)/(2*L)
    # Define the constants
    q_star = dl.Constant(q_star)
    #K = dl.Constant(K)
    eta = dl.Constant(eta_k)

    # Define discretization
    dt = T_max/N_steps

    # Define function space and boundaries
    mesh = dl.IntervalMesh(N, 0, L)
    P1 = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P2 = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    ME = dl.FunctionSpace(mesh, P1*P2)

    # Define functions
    u = dl.Function(ME)
    u0 = dl.Function(ME)
    u_test = dl.TestFunction(ME)

    h, q = dl.split(u)
    h0, q0 = dl.split(u0)
    v, p = dl.split(u_test)

    # Initial conditions
    u_init = InitialConditions(h1)
    u.interpolate(u_init)
    u0.interpolate(u_init)
    
    n = dl.FacetNormal(mesh)

    f0 = h*h/3 * dl.inner(dl.grad(q), dl.grad(p)) * dl.dx \
            - h/3 * q * dl.inner(dl.grad(h), dl.grad(p)) * dl.dx \
            + q * p * dl.dx \
            + h * dl.div(h * dl.Constant((1.0,))) * p * dl.dx \
            + h/3 * q * dl.inner(dl.grad(h), n) * p * dl.ds \
            - h*h/3 * dl.inner(dl.grad(q), n) * p *dl.ds

    f1 = (h - h0) * dl.Constant(eta) / dl.Constant(dt) * v * dl.dx \
          + dl.div(q * dl.Constant((1.0,))) * v * dl.dx

    f = f0 + f1
    time = 0.0
    count = 0
    h_data = []
    q_data = []
    tt = []
    def right_boundary(x, on_boundary):
        return x[0] > L - dl.DOLFIN_EPS

    def left_boundary(x, on_boundary):
        return x[0] < dl.DOLFIN_EPS
    
    xx = mesh.coordinates()
    while time < T_max:
        # Define the boundary conditions
        if time > 0:
            bcq = dl.DirichletBC(ME.sub(1), q_star, left_boundary)
        else:
            bcq = dl.DirichletBC(ME.sub(1), dl.Constant(0.0), left_boundary)

        bch = dl.DirichletBC(ME.sub(0), dl.Constant(h1), right_boundary)

        # Solve the nonlinear problem
        dl.solve(f == 0, u, bcs=[bch, bcq])
        
        h, q = dl.split(u)
        hh = np.zeros(xx.shape)
        qq = np.zeros(xx.shape)
        
        tt.append(time)
        for i in range(xx.shape[0]):
            hh[i] = h(xx[i])
            qq[i] = q(xx[i])
        h_data.append(hh.copy())
        q_data.append(qq.copy())
        
        if count % 10 == 0:
            print("Iteration: ", count, "time = ", time)
        
        time += dt
        count += 1
        u0.vector().set_local(u.vector().get_local())

    u = np.reshape(h_data,(count,len(xx)))
    s = np.reshape(q_data,(count,len(xx)))
    return u, s , xx, tt



def plot_fd_solution(xx, FD_sol, i, format_str="-"):
    plt.plot(xx, FD_sol[i,:], format_str, linewidth=2)
    plt.ylim([0, np.max(np.max(FD_sol))*1.2])

def plot_model_solution(xx, t, NN, format_str="-"):
    tt = np.ones(xx.shape) * t
    X = np.stack((xx, tt)).T
    u_pred, f_pred, _, _ = NN.predict(X)

    plt.plot(xx, u_pred, format_str, linewidth=2)
    plt.ylim([0, None])
    
def model_solution(xx, t, NN):
    u = []
    for ti in t:
          tt = np.ones(xx.shape) * ti
          X = np.stack((xx, tt)).T
          u_pred, f_pred, _, _ = NN.predict(X)
          u.append(u_pred[:,0])
    return np.reshape(u,(len(t),len(xx)))
    
def model_solution_di(xx, t, NN):
    u = []
    for ti in t:
          tt = np.ones(xx.shape) * ti
          X = np.stack((xx, tt)).T
          u_pred, q_pred, f1_pred, f2_pred, f_left, f_right = NN.predict(X)
          u.append(u_pred[:,0])
    return np.reshape(u,(len(t),len(xx)))
    
    
def plot_model_solution_di(xx, t, NN, format_str="-"):
    tt = np.ones(xx.shape) * t
    X = np.stack((xx, tt)).T
    u_pred, q_pred, f1_pred, f2_pred, f_left, f_right = NN.predict(X)
    plt.plot(xx, u_pred, format_str, linewidth=2)
    plt.ylim([0, None])

def plot_timestamps(x, dt, FD_sol, model, ind_tests, FD_fit=None):
    plt.rcParams.update({"font.size" : 16})
    for ind in ind_tests:
        plt.figure(figsize=(8,5))
        plot_fd_solution(x, FD_sol, ind, "-k")
        t_ind = ind * dt
        plot_model_solution(x, t_ind, model, "--r")

        if FD_fit is not None:
            plot_fd_solution(x, FD_fit, ind, "-b")

        plt.title("time = %g" %(t_ind))
        plt.xlabel("x")
        plt.ylabel("h")
        plt.grid(True, which="both")

        plt.legend(["PDE Solution with true values", "PINN", "PDE Solution with recovered values"], loc="lower right")
        plt.savefig("unsteady_figures/unsteady_bc_noiseless_res_%d.png" %(ind))
        plt.close()

def plot_animation(x, dt, FD_sol, model, ind_tests, FD_fit=None):
    for ind in ind_tests:
        plt.figure()
        plot_fd_solution(x, FD_sol, ind)
        t_ind = ind * dt
        plot_model_solution(x, t_ind, model)

        if FD_fit is not None:
            plot_fd_solution(x, FD_fit, ind)

        plt.title("time = %g" %(t_ind))
        plt.xlabel("x")
        plt.ylabel("h")
        plt.legend(["numerical", "NN", "numerical with fit K"])
        plt.pause(0.0001)
        plt.clf()
        plt.close()


def plot_animation_di(x, dt, FD_sol, model, ind_tests, FD_fit=None):
    for ind in ind_tests:
        plt.figure()
        plot_fd_solution(x, FD_sol, ind)
        t_ind = ind * dt
        plot_model_solution_di(x, t_ind, model)

        if FD_fit is not None:
            plot_fd_solution(x, FD_fit, ind)

        plt.title("time = %g" %(t_ind))
        plt.xlabel("x")
        plt.ylabel("h")
        plt.legend(["numerical", "NN", "numerical with fit K"])
        plt.pause(0.0001)
        plt.clf()
        plt.close()


