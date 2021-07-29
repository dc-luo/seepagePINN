import numpy as np 
import h5py 
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Select PDE model') 
parser.add_argument('-n', '--name', type=str, default="1mm", help="data file name") 
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

filename = args.name + ".h5"
print(filename)

out_file = h5py.File(filename, "r")
nq = out_file.get('n_total')[()]

X_data = np.array(out_file.get('X_data'))
u_data = np.array(out_file.get('u_data'))
X_test = np.array(out_file.get('X_test'))
u_test = np.array(out_file.get('u_test'))
training_list = out_file.get('training_list')[()]

K_truth = out_file.get('K_truth')[()]
K_dupuit = out_file.get("dupuit/K")[()]
K_dinucci = out_file.get("dinucci/K")[()]
print("Measured K from bead size: ", K_truth)
print("Inverted K (Dupuit): ", K_dupuit)
print("Inverted K (Dinucci): ", K_dinucci)

FIGSIZE = (8,6)
FIGDPI = 100
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 16
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['lines.linewidth'] = 3

n_color = 4
grad_color = plt.cm.get_cmap('turbo', 12)
base_color = 0.1
top_color = 0.9
color_incr = (top_color - base_color)/n_color

h_max = np.max(np.abs(u_test))
scale_q = 1e-4

if args.plot_prediction:
    for iq in range(nq):
        q = np.abs(slice_to_flow(X_test, iq, nq)[0,1]*scale_q)
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_test, iq, nq)[:,0], 'ok', label="Data")
        
        exponent = np.floor(np.log10(q))
        multiplier = q / 10**(exponent)

        # With collocation points
        groupname = "dinucci"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-',color=grad_color(0*color_incr+base_color), 
                label="DiNucci")

        groupname = "dupuit"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '--',color=grad_color(n_color*color_incr+base_color),
                label="Dupuit")
        K = out_file.get(groupname + "/K")[()]

        # With collocation points
        groupname = "none"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-', color='g', 
                label="None")


        if iq + 1 in training_list:
            plt.title(r"$q = %.3f \times 10^{%d}$ (Training)" %(multiplier, exponent))
        else:
            plt.title(r"$q = %.3f \times 10^{%d}$ (Test)" %(multiplier, exponent))
        plt.legend()
        plt.xlabel(r"$x$")
        plt.ylabel(r"$h$")
        plt.ylim([0, 1.1*h_max])
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig("figures/%s_prediction_%d.pdf" %(args.name, iq))


if args.plot_residual:
    for iq in range(nq):
        q = slice_to_flow(X_test, iq, nq)[0,1] * scale_q
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
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f_pred, iq, nq)[:,0], '-r', 
                label=r"$\alpha = %g$" %(alpha))
    
        plt.xlabel(r"$x$")
        plt.ylabel(r"PDE residual $f$")
        plt.title(r"$q = %g$" %(-q))
        plt.tight_layout()
        plt.savefig("figures/%s_residual_%d.pdf" %(args.name, iq))
    

L = 1.62
if args.plot_terms:
    for iq in range(nq):
        q = slice_to_flow(X_test, iq, nq)[0,1] * scale_q
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
        # No collocation points
        groupname = "dinucci"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f1_pred = -np.array(out_file.get(groupname + "/f1_pred"))
        f2_pred = -np.array(out_file.get(groupname + "/f2_pred"))
        f3_pred = -np.array(out_file.get(groupname + "/f3_pred"))
        K = out_file.get(groupname+"/K")[()]

        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(K*f1_pred, iq, nq)[:,0], '-', color=grad_color(1*color_incr),  label=r"$-\frac{K}{q}hh_x$")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f2_pred + f3_pred, iq, nq)[:,0], '-', color=grad_color(2*color_incr), label=r"$-\frac{1}{3}\partial_x(h h_x)$")
        # plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f2_pred, iq, nq)[:,0], '-', color=grad_color(2*color_incr), label=r"$-\frac{1}{3}h h_{xx}$")
        # plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f3_pred, iq, nq)[:,0], '-', color=grad_color(3*color_incr), label=r"$-\frac{1}{3} h_x^2$")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(K*f1_pred + f2_pred + f3_pred, iq, nq)[:,0], '--', color=grad_color(4*color_incr), label="Sum")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(X_test, iq, nq)[:,0]*0 + 1, '--k')
    
        pi = np.abs(2*q/(K* L))

        exponent = np.floor(np.log10(pi))
        multiplier = pi/10**(exponent)

        if iq + 1 in training_list:
            plt.title(r"$\Pi = %.3f \times 10^{%d}$ (Training)" %(multiplier, exponent))
        else:
            plt.title(r"$\Pi = %.3f \times 10^{%d}$ (Test)" %(multiplier, exponent))

        # plt.legend(loc='upper center', ncol=4) 
        plt.xlabel(r"$x$")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/%s_terms_%d.pdf" %(args.name, iq))

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
    plt.savefig("figures/%s_comparison_large.pdf" %(args.name))

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

    plt.savefig("figures/%s_comparison_small.pdf" %(args.name))

plt.show()
