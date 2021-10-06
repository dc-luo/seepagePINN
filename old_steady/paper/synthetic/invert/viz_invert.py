import numpy as np 
import h5py 
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Select PDE model') 
parser.add_argument('-m', '--flow_model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help="PDE choice for generating data: dinucci or dupuit")
parser.add_argument('-d', '--data_model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help="PDE choice for interpreting data: dinucci or dupuit")
parser.add_argument("-u", "--plot_prediction", help="plot model vs prediction as trends", action="store_true", default=False)
parser.add_argument("-f", "--plot_residual", help="plot model pde residual", action="store_true", default=False)
parser.add_argument("-c", "--plot_comparison", help="plot model vs prediction as scatter", action="store_true", default=False)
args = parser.parse_args()

L = 1

def slice_to_flow(arr, i, n):
    """ slice to the ith flow value given a total of n possible flow values""" 
    assert(arr.shape[0] % n == 0)
    incr = round(arr.shape[0]/n)
    i_lower = i * incr
    i_upper = (i+1) * incr
    return arr[i_lower:i_upper, :]

FIGSIZE = (12, 8)
FIGDPI = 100
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 16
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

data_model = args.data_model
flow_model = args.flow_model
filename = "data_" + data_model + "_" + flow_model + ".h5"

print(filename)

out_file = h5py.File(filename, "r")
qs = np.array(out_file.get('q'))
nq = qs.shape[0]

X_data = L - np.array(out_file.get('X_data'))
u_data = np.array(out_file.get('u_data'))
X_test = L - np.array(out_file.get('X_test'))
u_test = np.array(out_file.get('u_test'))

K_truth = out_file.get('K_truth')[()]
K_large = out_file.get('alpha_large/K')[()]
K_small = out_file.get('alpha_small/K')[()]
print("True K: ", K_truth)
print("With large alpha: ", K_large, " Error: ", (K_large-K_truth)/K_truth)
print("With small alpha: ", K_small, " Error: ", (K_small-K_truth)/K_truth)

if args.plot_prediction:
    for iq, q in enumerate(qs):
        plt.figure(figsize=(8,6), dpi=FIGDPI)
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_test, iq, nq)[:,0], 'ok', label="PDE")
        
        # No collocation points
        groupname = "alpha_large"
        alpha = out_file.get(groupname + "/alpha")[()]
        h_max = np.sqrt(alpha)
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-b', 
                label=r"$h_{max}^2$")
        K = out_file.get(groupname + "/K")[()]
        print(groupname, alpha, K)

        # With collocation points
        groupname = "alpha_small"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '--r', 
                label=r"$h_{max}^2/10$")
        plt.title("$q = %g$" %(q))
        plt.legend()
        K = out_file.get(groupname + "/K")[()]
        print(groupname, alpha, K)

        plt.xlabel(r"$x$")
        plt.ylabel(r"$h$")
        plt.ylim([0, h_max*1.1])
        plt.tight_layout()
        plt.savefig("figures/%s_invert_prediction_%d.pdf" %(args.data_model, iq))

    plt.show()

if args.plot_residual:
    for iq, q in enumerate(qs):
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
        # No collocation points
        groupname = "alpha_large"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f_pred, iq, nq)[:,0], '-b', 
                label=r"$\alpha = %g$" %(alpha))
    
        # With collocation points
        groupname = "alpha_small"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f_pred, iq, nq)[:,0], '--r', 
                label=r"$\alpha = %g$" %(alpha))
    
        plt.title(r"$q = %g$" %(q))
        plt.xlabel(r"$x$")
        plt.ylabel(r"PDE residual $f$")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/%s_residual_%d.pdf" %(args.data_model, iq))
    
    plt.show()

if args.plot_comparison:
    plt.figure(figsize=FIGSIZE, dpi=FIGDPI)

    groupname = "alpha_large"
    alpha = out_file.get(groupname + "/alpha")[()]
    u_pred = np.array(out_file.get(groupname + "/u_pred"))
    f_pred = np.array(out_file.get(groupname + "/f_pred"))
    plt.plot(u_test[:,0], u_pred[:,0], 'ob')

    plt.xlabel("Test data")
    plt.ylabel("Model prediction")
    plt.title(r"$\alpha = %g$" %(alpha))
    plt.xlim([0, None])
    plt.ylim([0, None])

    plt.tight_layout()
    plt.savefig("figures/%s_comparison_large.pdf" %(args.data_model))
    plt.figure(figsize=FIGSIZE, dpi=FIGDPI)

    groupname = "alpha_small"
    alpha = out_file.get(groupname + "/alpha")[()]
    u_pred = np.array(out_file.get(groupname + "/u_pred"))
    f_pred = np.array(out_file.get(groupname + "/f_pred"))
    plt.plot(u_test[:,0], u_pred[:,0], 'ob')

    plt.xlabel("Test data")
    plt.ylabel("Model prediction")
    plt.title(r"$\alpha = %g$" %(alpha))
    plt.xlim([0, None])
    plt.ylim([0, None])
    
    plt.tight_layout()
    plt.savefig("figures/%s_comparison_small.pdf" %(args.data_model))
    plt.show()
