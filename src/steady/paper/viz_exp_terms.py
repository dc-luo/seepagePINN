import os
import argparse

import numpy as np 
import h5py 
import matplotlib.pyplot as plt 

FIGSIZE = (8,6)
FIGDPI = 100

def configure_plot():
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 16
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['lines.linewidth'] = 3


def parse_args():
    parser = argparse.ArgumentParser(description='Select PDE model') 
    parser.add_argument('-n', "--name", type=str, default="1mm", help="data file name") 
    parser.add_argument("-i", "--index", type=int, default=0, help="Index of the data file")
    parser.add_argument("-u", "--plot_prediction", help="plot model vs prediction as trends", action="store_true", default=False)
    parser.add_argument("-f", "--plot_residual", help="plot model pde residual", action="store_true", default=False)
    parser.add_argument("-c", "--plot_comparison", help="plot model vs prediction as scatter", action="store_true", default=False)
    parser.add_argument("-t", "--plot_terms", help="plot the individual PDE terms", action="store_true", default=False)
    args = parser.parse_args()
    return args


def plot_prediction(X_test, u_test, q, out_file, grad_color, base_color, color_incr, path, savename, index):
    h_max = np.max(np.abs(u_test))
    plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
    plt.plot(X_test[:,0], u_test[:,0], 'ok', label="Data")
    
    exponent = np.floor(np.log10(q))
    multiplier = q / 10**(exponent)
    
    groupname = "dinucci"
    alpha = out_file.get(groupname + "/alpha")[()]
    u_pred = np.array(out_file.get(groupname + "/u_pred"))
    f_pred = np.array(out_file.get(groupname + "/f_pred"))
    plt.plot(X_test[:,0], u_pred[:,0], '-',color=grad_color(0*color_incr+base_color), 
            label="DiNucci")
    
    plt.title(r"$q = %.3f \times 10^{%d}$" %(multiplier, exponent))
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$h$")
    plt.ylim([0, 1.1*h_max])
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path + "figures/%s_prediction_%d.pdf" %(savename, index))


def plot_terms(X_test, q, out_file, grad_color, color_incr, path, savename, index):
    plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
    
    # Retrieve terms from data
    groupname = "dinucci"
    alpha = out_file.get(groupname + "/alpha")[()]
    u_pred = np.array(out_file.get(groupname + "/u_pred"))
    f1_pred = -np.array(out_file.get(groupname + "/f1_pred"))
    f2_pred = -np.array(out_file.get(groupname + "/f2_pred"))
    f3_pred = -np.array(out_file.get(groupname + "/f3_pred"))
    K = out_file.get(groupname+"/K")[()]
    
    # Plot individual terms
    plt.plot(X_test[:,0], K*f1_pred[:,0], '-', color=grad_color(1*color_incr),  label=r"$-\frac{K}{q}hh_x$")
    plt.plot(X_test[:,0], f2_pred[:,0] + f3_pred[:,0], '-', color=grad_color(2*color_incr), label=r"$-\frac{1}{3}\partial_x(h h_x)$")
    plt.plot(X_test[:,0], K*f1_pred[:,0] + f2_pred[:,0] + f3_pred[:,0], '--', color=grad_color(4*color_incr), label="Sum")
    plt.plot(X_test[:,0], np.ones(X_test[:,0].shape), '--k')
    
    # Compute nondimensional parameter
    L = 1.62
    pi = np.abs(2*q/(K* L))
    exponent = np.floor(np.log10(pi))
    multiplier = pi/10**(exponent)
    
    plt.title(r"$\Pi = %.3f \times 10^{%d}$" %(multiplier, exponent))
    plt.xlabel(r"$x$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + "figures/%s_terms_%d.pdf" %(savename, index))


def main():
    configure_plot()
    args = parse_args()

    save_path = "experimental/terms/"
    os.makedirs(save_path + "figures", exist_ok=True)
    filename = save_path + args.name + "%s" %(args.index) + ".h5"

    out_file = h5py.File(filename, "r")
    nq = out_file.get('n_total')[()]
    
    X_data = np.array(out_file.get('X_data'))
    u_data = np.array(out_file.get('u_data'))
    X_test = np.array(out_file.get('X_test'))
    u_test = np.array(out_file.get('u_test'))
    training_list = out_file.get('training_list')[()]
    print(filename)
    print(training_list)
    
    K_truth = out_file.get('K_truth')[()]
    K_dinucci = out_file.get("dinucci/K")[()]
    print("Measured K from bead size: ", K_truth)
    print("Inverted K (Dinucci): ", K_dinucci)

    n_color = 4
    grad_color = plt.cm.get_cmap('turbo', 12)
    base_color = 0.1
    top_color = 0.9
    color_incr = (top_color - base_color)/n_color
    
    q = np.abs(X_test[0,1])
    scale_q = 1e-4
    q = q * scale_q
    
    if args.plot_prediction:
        plot_prediction(X_test, u_test, q, out_file, grad_color, base_color, color_incr, save_path, args.name, args.index)
    
    if args.plot_terms:
        plot_terms(X_test, q, out_file, grad_color, color_incr, save_path, args.name, args.index)

    plt.show()

if __name__ == "__main__":
    main()
