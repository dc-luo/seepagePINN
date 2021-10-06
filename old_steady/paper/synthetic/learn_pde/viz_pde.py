import numpy as np 
import h5py 
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Select PDE model') 
parser.add_argument('-n', '--name', type=str, default="", help="Data file name, optional input")
parser.add_argument('-d', '--data_model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help="PDE choice for interpreting data: dinucci or dupuit")
parser.add_argument("-u", "--plot_prediction", help="plot model vs prediction as trends", action="store_true", default=False)
parser.add_argument("-f", "--plot_residual", help="plot model pde residual", action="store_true", default=False)
parser.add_argument("-c", "--plot_comparison", help="plot model vs prediction as scatter", action="store_true", default=False)
parser.add_argument("-t", "--plot_terms", help="plot the individual PDE terms", action="store_true", default=False)
args = parser.parse_args()

def slice_to_flow(arr, i, n):
    """ slice to the ith flow value given a total of n possible flow values""" 
    assert(arr.shape[0] % n == 0)
    incr = round(arr.shape[0]/n)
    i_lower = i * incr
    i_upper = (i+1) * incr
    return arr[i_lower:i_upper, :]

if args.name:
    filename = args.name + ".h5"
    print("Plotting ", args.name)
else:
    data_model = args.data_model
    filename = "data_" + data_model + ".h5"
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

FIGSIZE = (8, 6)
FIGDPI = 100
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 16
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['lines.linewidth'] = 3

n_color = 4
grad_color = plt.cm.get_cmap('turbo', 12)
base_color = 0.0
top_color = 0.8
color_incr = (top_color - base_color)/n_color


K = out_file.get("dinucci" + "/K")[()]
C2 = out_file.get("dinucci" + "/C2")[()]
C3 = out_file.get("dinucci" + "/C3")[()]
print(K, C2, C3)

if args.plot_prediction:
    for iq, q in enumerate(qs):
        plt.figure(figsize=FIGSIZE)
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_test, iq, nq)[:,0], 'ok', label="PDE")
        
        # No collocation points
        groupname = "dinucci"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        # plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-', color=grad_color(1*color_incr), label="DiNucci")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-b', label="DiNucci")
        K = out_file.get(groupname + "/K")[()]

        print(groupname, alpha, K)

        # With collocation points
        groupname = "dupuit"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        # plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-', color=grad_color(4*color_incr), label="Dupuit")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '--r', label="Dupuit")

        pi = 2*q/(K_truth * L)
        plt.title(r"$\Pi = %g$" %(pi))
        plt.xlabel(r"$x$")
        plt.ylabel(r"$h$")
        plt.legend()

        K = out_file.get(groupname + "/K")[()]
        print(groupname, alpha, K)
        plt.tight_layout()
        plt.savefig("figures/%s_prediction.pdf" %(filename[:-3]))

if args.plot_residual:
    for iq, q in enumerate(qs):
        plt.figure(figsize=FIGSIZE)
        # No collocation points
        groupname = "dinucci"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        # plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f_pred, iq, nq)[:,0], '-', color=grad_color(1*color_incr), label="DiNucci")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f_pred, iq, nq)[:,0], '-b', label="DiNucci")
    
        # With collocation points
        groupname = "dupuit"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        # plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f_pred, iq, nq)[:,0], '-', color=grad_color(4*color_incr), label="Dupuit")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f_pred, iq, nq)[:,0], '--r', label="Dupuit")
    
        pi = 2*q/(K_truth * L)
        plt.title(r"$\Pi = %g$" %(pi))
        plt.xlabel(r"$x$")
        plt.ylabel(r"$f$")
        plt.legend()
        plt.tight_layout()
    
        plt.savefig("figures/%s_residual.pdf" %(filename[:-3]))

if args.plot_terms:
    for iq, q in enumerate(qs):
        plt.figure(figsize=FIGSIZE)
        # No collocation points
        groupname = "dinucci"
        alpha = out_file.get(groupname + "/alpha")[()]
        C2 = out_file.get(groupname + "/C2")[()]
        C3 = out_file.get(groupname + "/C3")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f1_pred = -np.array(out_file.get(groupname + "/f1_pred"))
        f2_pred = -np.array(out_file.get(groupname + "/f2_pred"))
        f3_pred = -np.array(out_file.get(groupname + "/f3_pred"))


        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(K_truth*f1_pred, iq, nq)[:,0], '-', color=grad_color(1*color_incr),  label=r"$-\frac{K}{q}hh_x$")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(C2 * f2_pred, iq, nq)[:,0], '-', color=grad_color(2*color_incr), label=r"$-\frac{1}{3}h h_{xx}$")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(C3 * f3_pred, iq, nq)[:,0], '-', color=grad_color(3*color_incr), label=r"$-\frac{1}{3} h_x^2$")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(K_truth*f1_pred + C2*f2_pred + C3*f3_pred, iq, nq)[:,0], '--', color=grad_color(4*color_incr), label="Sum")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(X_test, iq, nq)[:,0]*0 + 1, '--k', linewidth=2)
    
        pi = 2*q/(K_truth * L)
        plt.xlabel(r"$x$")
        plt.ylabel(r"Term")
        plt.title(r"$\Pi = %g$" %(pi))
        # plt.legend(loc='lower center', ncol=4) 
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/%s_terms.pdf" %(filename[:-3]))


if args.plot_comparison:
    plt.figure(figsize=FIGSIZE)

    groupname = "dinucci"
    alpha = out_file.get(groupname + "/alpha")[()]
    u_pred = np.array(out_file.get(groupname + "/u_pred"))
    f_pred = np.array(out_file.get(groupname + "/f_pred"))
    plt.plot(u_test[:,0], u_pred[:,0], 'ob')

    plt.xlabel("Test data")
    plt.ylabel("Prediction")
    plt.title("DiNucci")
    # plt.xlim([0, None])
    # plt.ylim([0, None])
    plt.tight_layout()

    plt.savefig("figures/%s_comparison_dinucci.pdf" %(filename[:-3]))

    plt.figure(figsize=FIGSIZE, dpi=FIGDPI)

    groupname = "dupuit"
    alpha = out_file.get(groupname + "/alpha")[()]
    u_pred = np.array(out_file.get(groupname + "/u_pred"))
    f_pred = np.array(out_file.get(groupname + "/f_pred"))
    plt.plot(u_test[:,0], u_pred[:,0], 'ob')

    plt.xlabel("Test data")
    plt.ylabel("Prediction")
    plt.title("Dupuit")
    # plt.xlim([0, None])
    # plt.ylim([0, None])

    plt.tight_layout()
    plt.savefig("figures/%s_comparison_dupuit.pdf" %(filename[:-3]))



plt.show()
