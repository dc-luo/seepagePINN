import os
import glob 
import argparse

import numpy as np 
import h5py 
import matplotlib.pyplot as plt 


FIGSIZE = (8, 6)
FIGDPI = 100

def parse_args():
    parser = argparse.ArgumentParser(description='Select PDE model') 
#   parser.add_argument('-n', '--name', type=str, default="", help="Data file name, optional input")
#     parser.add_argument('-d', '--data_model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help="PDE choice for interpreting data: dinucci or dupuit")
#     parser.add_argument("-f", "--plot_residual", help="plot model pde residual", action="store_true", default=False)
#   parser.add_argument("-c", "--plot_comparison", help="plot model vs prediction as scatter", action="store_true", default=False)
    parser.add_argument("-u", "--plot_prediction", help="plot model vs prediction as trends", action="store_true", default=False)
    parser.add_argument("-t", "--plot_terms", help="plot the individual PDE terms", action="store_true", default=False)
    args = parser.parse_args()
    return args

def slice_to_flow(arr, i, n):
    """ slice to the ith flow value given a total of n possible flow values""" 
    assert(arr.shape[0] % n == 0)
    incr = round(arr.shape[0]/n)
    i_lower = i * incr
    i_upper = (i+1) * incr
    return arr[i_lower:i_upper, :]

def set_figure_properties():
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 16
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['lines.linewidth'] = 3

def plot_terms(filename, save_dir, args):
    """ Plot the terms given an input data name""" 
    print("Plotting ", filename)
    
    out_file = h5py.File(filename, "r")
    qs = np.array(out_file.get('q'))
    nq = qs.shape[0]
    
    X_data = np.array(out_file.get('X_data'))
    u_data = np.array(out_file.get('u_data'))
    X_test = np.array(out_file.get('X_test'))
    u_test = np.array(out_file.get('u_test'))
    
    K_truth = out_file.get('K_truth')[()]
    L = 1.0
    
    X_data = L - X_data
    X_test = L - X_test
    
    n_color = 4
    grad_color = plt.cm.get_cmap('turbo', 12)
    base_color = 0.0
    top_color = 0.8
    color_incr = (top_color - base_color)/n_color
    
    K_dinucci = out_file.get("dinucci" + "/K")[()]
    K_dupuit = out_file.get("dinucci" + "/K")[()]

    for iq, q in enumerate(qs):
        plt.figure(figsize=FIGSIZE)
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_test, iq, nq)[:,0], 'ok', label="PDE")
        
        # No collocation points
        groupname = "dinucci"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-b', label="Di Nucci")
        K = out_file.get(groupname + "/K")[()]
        pi = 2*q/(K_truth * L)
        plt.xlabel(r"$x$")
        plt.ylabel(r"Term")
        plt.title(r"$\Pi = %g$" %(pi))
        plt.savefig(save_dir + "/%g_prediction.pdf" %(K_truth))

    for iq, q in enumerate(qs):
        plt.figure(figsize=FIGSIZE)
        # No collocation points
        groupname = "dinucci"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f1_pred = -np.array(out_file.get(groupname + "/f1_pred"))
        f2_pred = -np.array(out_file.get(groupname + "/f2_pred"))
        f3_pred = -np.array(out_file.get(groupname + "/f3_pred"))

        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(K_truth*f1_pred, iq, nq)[:,0], 
                '-', color=grad_color(1*color_incr),  label=r"$-\frac{K}{q}hh_x$")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f2_pred + f3_pred, iq, nq)[:,0], 
                '-', color=grad_color(2*color_incr), label=r"$-\frac{1}{3}\partial_x(h h_x)$")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(K_truth*f1_pred + f2_pred + f3_pred, iq, nq)[:,0], 
                '--', color=grad_color(4*color_incr), label="Sum")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(X_test, iq, nq)[:,0]*0 + 1, 
                '--k', linewidth=2)
    
        pi = 2*q/(K_truth * L)
        plt.xlabel(r"$x$")
        plt.ylabel(r"Term")
        plt.title(r"$\Pi = %g$" %(pi))
        # plt.legend(loc='lower center', ncol=4) 
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir + "/%g_terms.pdf" %(K_truth))

    plt.show()

def main():
    args = parse_args()
    data_names = glob.glob("synthetic/terms/*.h5")
    set_figure_properties()
    save_dir = "synthetic/terms/figures"
    os.makedirs(save_dir, exist_ok=True)
    for name in data_names: 
        plot_terms(name, save_dir, args)

if __name__ == "__main__":
    main()

