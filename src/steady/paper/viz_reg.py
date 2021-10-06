import os 

import numpy as np 
import h5py 
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Select PDE model') 
parser.add_argument('-n', '--name', type=str, default="", help="Data file name, optional input")
parser.add_argument('-m', '--model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help='PDE choice: dinucci or dupuit')
parser.add_argument("-u", "--plot_prediction", help="plot model vs prediction as trends", action="store_true", default=False)
parser.add_argument("-c", "--plot_comparison", help="plot model vs prediction as scatter", action="store_true", default=False)
parser.add_argument("-l", "--plot_lcurve", help="plot L-curve analysis", action="store_true", default=False)
args = parser.parse_args()

path = "synthetic/regularization/"

os.makedirs(path + "figures", exist_ok=True)

if args.name:
    filename = path + args.name + ".h5"
    print("Plotting ", args.name)
else:
    flow_model = args.model
    filename = path + "data_" + flow_model + ".h5"
    print("Plotting ", flow_model, " results")

def slice_to_flow(arr, i, n):
    """ slice to the ith flow value given a total of n possible flow values""" 
    assert(arr.shape[0] % n == 0)
    incr = round(arr.shape[0]/n)
    i_lower = i * incr
    i_upper = (i+1) * incr
    return arr[i_lower:i_upper, :]

L = 1.0

output_training = h5py.File(filename, "r")
qs = np.array(output_training.get('q'))
nq = qs.shape[0]
n_alpha = output_training.get('n_alpha')[()]

print("N ALPHA:", n_alpha)

X_data = np.array(output_training.get('X_data'))
u_data = np.array(output_training.get('u_data'))
X_test = np.array(output_training.get('X_test'))
u_test = np.array(output_training.get('u_test'))

FIGSIZE = (12, 8)
FIGDPI = 100
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 16
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

grad_color = plt.cm.get_cmap('turbo', 12)
base_color = 0.0
top_color = 0.8
color_incr = (top_color - base_color)/n_alpha

h_max = np.max(u_test[:,0])

if args.plot_prediction:
    for iq, q in enumerate(qs):
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
        plt.plot(L - slice_to_flow(X_data, iq, nq)[:,0], slice_to_flow(u_data, iq, nq)[:,0], 'ok', label="Data")
        plt.plot(L - slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_test, iq, nq)[:,0], '--k', label="PDE")
    
        for i_alpha in range(n_alpha):
            groupname = "alpha_%s" %(i_alpha)
            alpha = output_training.get(groupname + "/alpha")[()]
            u_pred = np.array(output_training.get(groupname + "/u_pred"))
            f_pred = np.array(output_training.get(groupname + "/f_pred"))

            if alpha >= np.finfo(float).eps: 
                log_alpha = np.log10(alpha)
    
                plt.plot(L - slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-', 
                        color=grad_color(base_color + i_alpha * color_incr), label=r"$\alpha = 10^{%g}$" %(log_alpha))
            else:
                plt.plot(L - slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-', 
                        color=grad_color(base_color + i_alpha * color_incr), label=r"$\alpha = 0$")
    
        plt.xlabel(r"$x$")
        plt.ylabel(r"$h$")
        plt.ylim([0, h_max*1.1])
        plt.title(r"$q = %g$" %(q))
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + "figures/%s_prediction_%d.pdf" %(args.model, iq))
    
    plt.show()

if args.plot_comparison:
    for i_alpha in range(n_alpha):
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
        groupname = "alpha_%s" %(i_alpha)
        alpha = output_training.get(groupname + "/alpha")[()]
        u_pred = np.array(output_training.get(groupname + "/u_pred"))
        f_pred = np.array(output_training.get(groupname + "/f_pred"))
    
        plt.plot(u_test[:,0], u_pred[:,0], 'ob') 
        if alpha > np.finfo(float).eps:
            log_alpha = np.log10(alpha)
            plt.title(r"$\alpha = 10^{%g}$" %(log_alpha))
        else:
            plt.title(r"$\alpha = 0$")

        plt.xlabel("Test data")
        plt.ylabel("Model prediction")

        plt.tight_layout()
        plt.savefig(path + "figures/%s_comparison_%d.pdf" %(args.model, i_alpha))
    
    plt.show()
    
if args.plot_lcurve:
    plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
    data_misfits = []
    pde_misfits = [] 

    for i_alpha in range(1,n_alpha):
        groupname = "alpha_%s" %(i_alpha)
        alpha = output_training.get(groupname + "/alpha")[()]
        u_train = np.array(output_training.get(groupname + "/u_train"))
        f_train = np.array(output_training.get(groupname + "/f_train"))

        data_misfit = np.linalg.norm(u_train[:,0] - u_data[:,0])
        pde_misfit = np.linalg.norm(f_train[:,0]) 

        data_misfits.append(data_misfit)
        pde_misfits.append(pde_misfit)

        plt.loglog(data_misfit, pde_misfit, 'o', color=grad_color(base_color + i_alpha * color_incr))
        
    print("Theoretically useful regularization parameter. Mean: ", np.mean(u_data)**2, " Max: ", np.max(u_data)**2)
    plt.loglog(data_misfits, pde_misfits, ':k')

    plt.grid(True)
    plt.xlabel(r'Data misfit $\|h_{NN} - h_{data}\|$')
    plt.ylabel(r'PDE Misfit $\|f_{NN}\|$')
    plt.tight_layout()
    plt.savefig(path + "figures/%s_lcurve.pdf" %(args.model))
    plt.show()
