import os 
import sys

import tensorflow as tf 
import numpy as np 
import h5py 
import argparse
import matplotlib.pyplot as plt 

sys.path.append("../")
from seepagePINN import *

def plot_scatter(model, X_test, u_test):
    u_pred_test, _ = model.predict(X_test)
    ax = plt.axes(projection='3d')
    ax.view_init(22, 45)
    ax.scatter3D(X_test[:,0], X_test[:,1], u_test[:,0], color='r')
    ax.scatter3D(X_test[:,0], X_test[:,1], u_pred_test[:,0], color='b')
    plt.xlabel("x")
    plt.ylabel("q")

def evaluate_K(model):
    return np.exp(model.sess.run(model.lambda_1)[0])

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

parser = argparse.ArgumentParser(description='Select PDE model') 
parser.add_argument("-c", "--case", type=str, default="1mm", help="data case")
parser.add_argument("-u", "--plot_prediction", help="plot model vs prediction as trends", action="store_true", default=False)
parser.add_argument("-s", "--plot_scatter", help="Plot the prediction vs test as scatter", action="store_true", default=False)
parser.add_argument("-f", "--plot_residual", help="plot residuals", action="store_true", default=False)
args = parser.parse_args()

FIGSIZE = (8, 6)
FIGDPI = 100
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 16
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


path = "experimental/all_models/%s/" %(args.case)
run_data = h5py.File(path + "data.h5", "r") 

X_train = np.array(run_data.get("X_data"))
u_train = np.array(run_data.get("u_data"))
X_test = np.array(run_data.get("X_test"))
u_test = np.array(run_data.get("u_test"))
nq = run_data.get("N_sets")[()]
scale_q = run_data.get("scale_q")[()]
alpha = run_data.get("alpha")[()]
L = run_data.get("L")[()]
X_colloc = None

training_list = np.array(run_data.get("training_list"), dtype=int)
layers = np.array(run_data.get("layers"), dtype=int)
K_truth = run_data.get("K_truth")[()]

grad_color = plt.cm.get_cmap('turbo', 12)
base_color = 0.0
top_color = 1.0
n_color = 5
color_incr = (top_color-base_color)/n_color

os.makedirs(path + "/figures", exist_ok=True)


dupuit_file = h5py.File(path + "dupuit_fit.h5", "r")
K_dupuit = dupuit_file.get("K")[()]
dupuit_file.close()

dinucci_file = h5py.File(path + "dinucci_fit.h5", "r")
K_dinucci = dinucci_file.get("K")[()]
dinucci_file.close()

print("K measured: ", K_truth)
print("K Dupuit: ", K_dupuit) 
print("K dinucci: ", K_dinucci)


groupnames = ["dupuit_fit", "dinucci_fit", "dupuit_flow", "dinucci_flow", "vanilla"]
labelnames = ["Dupuit (Inverted K)", "Di Nucci (Inverted K)", "Dupuit (Fixed K)", "Di Nucci (Fixed K)", "Plain NN"]
linetypes = ["-", "-", "--", "--", "-"]
h_max = np.max(u_test[:,0])

if args.plot_prediction:
    for iq in range(nq):
        q = slice_to_flow(X_test, iq, nq)[0, 1]*scale_q
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_test, iq, nq)[:,0], 'ok', label="Data")
    
        for i_group, groupname in enumerate(groupnames):
            output_file = h5py.File(path + groupname + ".h5", "r") 
            u_pred = np.array(output_file.get("u_pred"))
            output_file.close()
            plt.plot(slice_to_flow(X_test, iq, nq)[:,0], 
                    slice_to_flow(u_pred, iq, nq)[:,0],
                    linetypes[i_group],
                    color=grad_color(base_color + i_group * color_incr), 
                    label=labelnames[i_group])
        
        factor, exponent = exponential_form(np.abs(q))

        if iq in training_list:
            plt.title(r"$q = %.2f \times 10^{%d} (\mathrm{m}^2/\mathrm{s})$ (Training)" %(factor, exponent))
            # plt.title(r"$q = %g$ (Training)" %(-q))
        else:
            plt.title(r"$q = %.2f \times 10^{%d} (\mathrm{m}^2/\mathrm{s})$ (Test)" %(factor, exponent))
            # plt.title(r"$q = %g$ (Test)" %(-q))

        plt.xlabel(r"$x$ (m)")
        plt.ylabel(r"$h$ (m)")
        plt.ylim([0, h_max*1.1])
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(path + "figures/%s_prediction_%d.pdf" %(args.case, iq))
    plt.show()


if args.plot_residual:
    for iq in range(nq):
        q = slice_to_flow(X_test, iq, nq)[0, 1]*scale_q
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
    
        for i_group, groupname in enumerate(groupnames):
            output_file = h5py.File(path + groupname + ".h5", "r") 
            f_pred = np.array(output_file.get("f_pred"))
            output_file.close()
            plt.semilogy(slice_to_flow(X_test, iq, nq)[:,0], 
                    slice_to_flow(np.abs(f_pred), iq, nq)[:,0],
                    linetypes[i_group],
                    color=grad_color(base_color + i_group * color_incr), 
                    label=labelnames[i_group])
        
        factor, exponent = exponential_form(np.abs(q))

        if iq in training_list:
            plt.title(r"$q = %.2f \times 10^{%d} (\mathrm{m}^2/\mathrm{s})$ (Training)" %(factor, exponent))
            # plt.title(r"$q = %g$ (Training)" %(-q))
        else:
            plt.title(r"$q = %.2f \times 10^{%d} (\mathrm{m}^2/\mathrm{s})$ (Test)" %(factor, exponent))
            # plt.title(r"$q = %g$ (Test)" %(-q))

        plt.xlabel(r"$x$ (m)")
        plt.ylabel(r"$f$")
        # plt.ylim([0, h_max*1.1])
        plt.legend(loc="upper left")
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.savefig(path + "figures/%s_residual_%d.pdf" %(args.case, iq))
    plt.show()


grey = [0.5, 0.5, 0.5]
if args.plot_scatter:
    for labelname, groupname in zip(labelnames, groupnames):
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
        
        output_file = h5py.File(path + groupname + ".h5", "r") 
        u_pred = np.array(output_file.get("u_pred"))
        output_file.close()

        u_pred_plot = np.array([])
        u_test_plot = np.array([]) 

        for iq in range(nq):
            if iq not in training_list:
                u_pred_plot = np.append(u_pred_plot, slice_to_flow(u_pred, iq, nq))
                u_test_plot = np.append(u_test_plot, slice_to_flow(u_test, iq, nq))
        
        r_squared = np.corrcoef(u_test_plot, u_pred_plot)[0,1]**2
        rmse = np.sqrt(np.mean((u_pred_plot-u_test_plot)**2))
        rmse_data = np.sqrt(np.mean((u_test_plot)**2))

        plt.plot(u_test_plot, u_pred_plot, 'o', color=grey)
        # plt.text(0.08, 0.2, r"$R^2 = %.3f$" %(r_squared))
        plt.text(0, 0.8*np.max(u_pred_plot), r"$RMSE = %.3g$" %(rmse))
        plt.xlabel("Test data")
        plt.ylabel("Predictions for test data") 
        plt.title(labelname)
        plt.axis("equal")
        # plt.xlim([0, None])
        plt.tight_layout()
        plt.savefig(path + "figures/%s_scatter_%s.pdf" %(args.case,groupname))
    plt.show()
        
