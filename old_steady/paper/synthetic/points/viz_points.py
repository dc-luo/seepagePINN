import numpy as np 
import h5py 
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Select PDE model') 
parser.add_argument('-m', '--model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help='PDE choice: dinucci or dupuit')
parser.add_argument("-u", "--plot_prediction", help="plot model vs prediction as trends", action="store_true", default=False)
parser.add_argument("-f", "--plot_residual", help="plot PDE residual", action="store_true", default=False)
parser.add_argument("-c", "--plot_comparison", help="plot model vs prediction as scatter", action="store_true", default=False)
parser.add_argument("-l", "--plot_lcurve", help="plot L-curve analysis", action="store_true", default=False)
args = parser.parse_args()

def flip_x(x, L):
    return L - x

def slice_to_flow(arr, i, n):
    """ slice to the ith flow value given a total of n possible flow values""" 
    assert(arr.shape[0] % n == 0)
    incr = round(arr.shape[0]/n)
    i_lower = i * incr
    i_upper = (i+1) * incr
    return arr[i_lower:i_upper, :]

flow_model = args.model

print("Plotting ", flow_model, " results")

filename = "data_" + flow_model + ".h5"
print(filename)

output_training = h5py.File(filename, "r")
qs = np.array(output_training.get('q_data'))
nq = qs.shape[0]
n_runs = output_training.get('n_runs')[()]

X_data = np.array(output_training.get('X_data'))
u_data = np.array(output_training.get('u_data'))
X_test = np.array(output_training.get('X_test'))
u_test = np.array(output_training.get('u_test'))

L = 1

FIGSIZE = (12, 8)
FIGDPI = 100
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 16
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

grad_color = plt.cm.get_cmap('turbo', 12)
base_color = 0.0
top_color = 0.8
color_incr = (top_color - base_color)/n_runs

if args.plot_prediction:
    for iq, q in enumerate(qs):
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
        plt.plot(flip_x(slice_to_flow(X_data, iq, nq)[:,0], L), slice_to_flow(u_data, iq, nq)[:,0], 'ok', label="Data")
        plt.plot(flip_x(slice_to_flow(X_test, iq, nq)[:,0], L), slice_to_flow(u_test, iq, nq)[:,0], '--k', label="PDE")
    
        for i_colloc in range(n_runs):
            groupname = "n_colloc_%d" %(i_colloc)
            n_colloc = output_training.get(groupname + "/n_colloc")[()]
            u_pred = np.array(output_training.get(groupname + "/u_pred"))
    
            plt.plot(flip_x(slice_to_flow(X_test, iq, nq)[:,0], L), slice_to_flow(u_pred, iq, nq)[:,0], '-', 
                    color=grad_color(base_color + i_colloc * color_incr), label=r"$N_c = %g$" %(n_colloc))
    
        plt.title(r"$q = %g$" %(q))
        plt.xlabel("$x$")
        plt.ylabel("$h$")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/%s_prediction_%d.pdf" %(args.model, iq))
    
    plt.show()


if args.plot_residual:
    for iq, q in enumerate(qs):
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)

        for i_colloc in range(1, n_runs):
            groupname = "n_colloc_%d" %(i_colloc)
            n_colloc = output_training.get(groupname + "/n_colloc")[()]
            f_pred = np.abs(np.array(output_training.get(groupname + "/f_pred")))
    
            plt.semilogy(flip_x(slice_to_flow(X_test, iq, nq)[:,0], L), slice_to_flow(f_pred, iq, nq)[:,0], '-', 
                    color=grad_color(base_color + i_colloc * color_incr), label=r"$N_c = %g$" %(n_colloc))
    
        plt.xlabel("$x$")
        plt.ylabel("PDE residual $f$")
        plt.title(r"$q = %g$" %(q))
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/%s_residual_%d.pdf" %(args.model, iq))
    
    plt.show()


if args.plot_comparison:
    for i_colloc in range(n_runs):
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
        groupname = "n_colloc_%d" %(i_colloc)
        n_colloc = output_training.get(groupname + "/n_colloc")[()]
        u_pred = np.array(output_training.get(groupname + "/u_pred"))
        f_pred = np.array(output_training.get(groupname + "/f_pred"))
    
        plt.plot(u_test[:,0], u_pred[:,0], 'ob') 
        plt.title(r"$N_c = %g$" %(n_colloc))
        plt.tight_layout()
        plt.savefig("figures/%s_comparison_%d.pdf" %(args.model, i_colloc))
    
    plt.show()
    
if args.plot_lcurve:
    plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
    data_misfits = []
    pde_misfits = [] 

    for i_colloc in range(1, n_runs):
        groupname = "n_colloc_%d" %(i_colloc)
        n_colloc = output_training.get(groupname + "/n_colloc")[()]
        u_train = np.array(output_training.get(groupname + "/u_train"))
        f_train = np.array(output_training.get(groupname + "/f_train"))

        data_misfit = np.linalg.norm(u_train[:,0] - u_data[:,0])
        pde_misfit = np.linalg.norm(f_train[:,0]) 

        data_misfits.append(data_misfit)
        pde_misfits.append(pde_misfit)

        plt.loglog(data_misfit, pde_misfit, 'o', color=grad_color(base_color + i_colloc * color_incr))
        
    plt.grid(True)
    plt.xlabel(r'Data misfit $\|h_{NN} - h_{data}\|$')
    plt.ylabel(r'PDE Misfit $\|f_{NN}\|$')
    plt.tight_layout()
    plt.savefig("%s_lcurve.pdf" %(args.model))
    plt.show()
