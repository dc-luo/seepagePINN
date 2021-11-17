import os 
import sys

import tensorflow as tf 
import numpy as np 
import h5py 
import argparse
import matplotlib.pyplot as plt 

sys.path.append("../")
from seepagePINN import *

FIGSIZE = (8, 6)
FIGDPI = 100
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 16
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

def slice_to_flow(arr, i, n):
    """ slice to the ith flow value given a total of n possible flow values""" 
    assert(arr.shape[0] % n == 0)
    incr = round(arr.shape[0]/n)
    i_lower = i * incr
    i_upper = (i+1) * incr
    return arr[i_lower:i_upper, :]

def exponential_form(q):
    """ returns q as A*10^(B) """ 
    logq = np.log10(q)
    exponent = np.floor(logq)
    factor = q / 10**(exponent) 
    return factor, exponent 

def parse_args():
    parser = argparse.ArgumentParser(description='Select PDE model') 
    parser.add_argument('-m', '--flow_model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help="PDE choice for generating data: dinucci or dupuit")
    parser.add_argument('-d', '--data_model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help="PDE choice for interpreting data: dinucci or dupuit")
    parser.add_argument("-c", "--case", type=str, default="1mm", help="data case")
    parser.add_argument("-u", "--plot_prediction", help="plot model vs prediction as trends", action="store_true", default=False)
    parser.add_argument("-f", "--plot_residual", help="Plot the pde residual", action="store_true", default=False)
    args = parser.parse_args()
    return args

def plot_prediction(q_list, X_test, u_test, data_file, grad_color, base_color, color_incr, path, savename):
    # alpha_reference = np.mean(u_test**2)
    alpha_reference = data_file.get("alpha_reference")[()]
    h_max = np.max(u_test)
    nq = len(q_list) 
    groupnames = ["alpha_small", "alpha_medium", "alpha_large"]
    for iq in range(nq):
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_test, iq, nq)[:,0], 'ok', label="Data")

        for i_group, groupname in enumerate(groupnames):
            u_pred = np.array(data_file.get("%s/u_pred" %(groupname)))
            alpha = data_file.get("%s/alpha" %(groupname))[()]
            factor, exponent = exponential_form(alpha/alpha_reference)
            labelname = r"$%g \bar{\alpha}$" %(alpha/alpha_reference)

            plt.plot(slice_to_flow(X_test, iq, nq)[:,0], 
                    slice_to_flow(u_pred, iq, nq)[:,0],
                    color=grad_color(base_color + i_group * color_incr), 
                    label=labelname)

        factor, exponent = exponential_form(np.abs(q_list[iq]))

        plt.title(r"$q = %.2f \times 10^{%d}$" %(factor, exponent))

        plt.xlabel(r"$x$")
        plt.ylabel(r"$h$")
        plt.ylim([0, h_max*1.1])
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(path + "figures/%s_prediction_%d.pdf" %(savename, iq))

def plot_residual(q_list, X_test, u_test, data_file, grad_color, base_color, color_incr, path, savename):
    alpha_reference = data_file.get("alpha_reference")[()]
    h_max = np.max(u_test)
    nq = len(q_list) 
    groupnames = ["alpha_small", "alpha_medium", "alpha_large"]
    # labelnames = ["Large", "Small"]
    for iq in range(nq):
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)

        for i_group, groupname in enumerate(groupnames):
            f_pred = np.array(data_file.get("%s/f_pred" %(groupname)))
            alpha = data_file.get("%s/alpha" %(groupname))
            factor, exponent = exponential_form(alpha)
            # labelname = r"$10^{%d} \bar{\alpha}$" %(exponent)
            labelname = r"$%g \bar{\alpha}$" %(alpha/alpha_reference)
            plt.semilogy(slice_to_flow(X_test, iq, nq)[:,0], 
                    slice_to_flow(np.abs(f_pred), iq, nq)[:,0],
                    color=grad_color(base_color + i_group * color_incr), 
                    label=labelname)

        factor, exponent = exponential_form(np.abs(q_list[iq]))

        plt.title(r"$q = %.2f \times 10^{%d}$" %(factor, exponent))

        plt.xlabel(r"$x$")
        plt.ylabel(r"$|f_{NN}|$")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(path + "figures/%s_residual_%d.pdf" %(savename, iq))

def main():
    args = parse_args()
    path = "synthetic/invert/"
    run_name = "%s_%s" %(args.flow_model, args.data_model)
    run_data = h5py.File(path + "data_%s.h5" %(run_name), "r")
    
    L = 1
    X_train = np.array(run_data.get("X_data"))
    u_train = np.array(run_data.get("u_data"))
    X_test = np.array(run_data.get("X_test"))
    u_test = np.array(run_data.get("u_test"))

    X_train = L - X_train
    X_test = L - X_test
    # scale_q = run_data.get("scale_q")[()]
    
    q_list = np.array(run_data.get("q"))
    training_list = np.array(run_data.get("training_list"), dtype=int)
    K_truth = run_data.get("K_truth")[()]
    
    grad_color = plt.cm.get_cmap('turbo', 12)
    base_color = 0.2
    top_color = 1.0
    n_color = 2
    color_incr = (top_color-base_color)/n_color
    
    os.makedirs(path + "/figures", exist_ok=True)
    groupnames = ["alpha_large", "alpha_small"]
    K_large = run_data.get("alpha_large/K")[()]
    K_medium = run_data.get("alpha_medium/K")[()]
    K_small = run_data.get("alpha_small/K")[()]
    
    print("Flow model: ", args.flow_model)
    print("Data model: ", args.data_model)
    print("K (truth): ", K_truth)
    print("K (large): ", K_large)
    print("K (medium): ", K_medium)
    print("K (small): ", K_small)
    
    if args.plot_prediction:
        plot_prediction(q_list, X_train, u_train, run_data, grad_color, base_color, color_incr, path, run_name)

    if args.plot_residual:
        plot_residual(q_list, X_train, u_train, run_data, grad_color, base_color, color_incr, path, run_name)

    plt.show()

if __name__ == "__main__":
    main()
