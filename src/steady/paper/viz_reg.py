import os 

import numpy as np 
import h5py 
import argparse
import matplotlib.pyplot as plt 
import matplotlib.ticker

FIGSIZE = (12, 8)
FIGDPI = 100
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 24
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

def get_exponent(a):
    exponent = np.floor(np.log10(a))
    significand = a/10**(exponent)
    return significand, exponent

def parse_args():
    parser = argparse.ArgumentParser(description='Select PDE model') 
    parser.add_argument('-n', '--name', type=str, default="", help="Data file name, optional input")
    parser.add_argument('-m', '--model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help='PDE choice: dinucci or dupuit')
    parser.add_argument("-u", "--plot_prediction", help="plot model vs prediction as trends", action="store_true", default=False)
    parser.add_argument("-f", "--plot_residual", help="plot model vs prediction as trends", action="store_true", default=False)
    parser.add_argument("-c", "--plot_comparison", help="plot model vs prediction as scatter", action="store_true", default=False)
    parser.add_argument("-l", "--plot_lcurve", help="plot L-curve analysis", action="store_true", default=False)
    parser.add_argument("--legend", help="add legend for plots", action="store_true", default=False)

    args = parser.parse_args()
    return args

def slice_to_flow(arr, i, n):
    """ slice to the ith flow value given a total of n possible flow values""" 
    assert(arr.shape[0] % n == 0)
    incr = round(arr.shape[0]/n)
    i_lower = i * incr
    i_upper = (i+1) * incr
    return arr[i_lower:i_upper, :]


def plot_prediction(output_training, grad_color, base_color, color_incr, path, savename, legend=False, subselection=None):
    L = 1.0
    X_data = np.array(output_training.get('X_data'))
    u_data = np.array(output_training.get('u_data'))
    X_test = np.array(output_training.get('X_test'))
    u_test = np.array(output_training.get('u_test'))
    q_list = np.array(output_training.get('q'))
    n_alpha = output_training.get("n_alpha")[()]
    nq = len(q_list)

    h_max = np.max(u_test[:,0])
    alpha_reference = output_training.get('alpha_reference')[()]

    if subselection is None:
        alpha_list = range(n_alpha)
    else:
        alpha_list = subselection


    for iq, q in enumerate(q_list):
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
        plt.plot(L - slice_to_flow(X_data, iq, nq)[:,0], slice_to_flow(u_data, iq, nq)[:,0], 'ok', label="Data")
        plt.plot(L - slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_test, iq, nq)[:,0], '--k', label="PDE")
    
        for i_alpha in alpha_list:
            groupname = "alpha_%s" %(i_alpha)
            alpha = output_training.get(groupname + "/alpha")[()]
            u_pred = np.array(output_training.get(groupname + "/u_pred"))
            f_pred = np.array(output_training.get(groupname + "/f_pred"))

            if alpha >= np.finfo(float).eps: 
                significand, exponent = get_exponent(alpha/alpha_reference) 
                plt.plot(L - slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-', linewidth=2,
                        color=grad_color(base_color + i_alpha * color_incr), label=r"$\alpha = %g \times 10^{%g} \bar{\alpha}$" %(significand, exponent))
                        # color=grad_color(base_color + i_alpha * color_incr), label=r"$\alpha = 10^{%g}$" %(log_alpha))
            else:
                plt.plot(L - slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-', linewidth=2,
                        color=grad_color(base_color + i_alpha * color_incr), label=r"$\alpha = 0$")
    
        plt.xlabel(r"$x \; (\mathrm{m})$")
        plt.ylabel(r"$h \; (\mathrm{m})$")
        plt.ylim([0, h_max*1.1])
        sq, eq = get_exponent(q)
        plt.title(r"$q = %.2f \times 10^{%d} \; (\mathrm{m}^2/\mathrm{s})$" %(sq, eq))
        # plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
        if legend:
            plt.legend()

        plt.tight_layout()

        if legend:
            plt.savefig(path + "figures/%s_prediction_%d_legend.pdf" %(savename, iq))
        else:
            plt.savefig(path + "figures/%s_prediction_%d.pdf" %(savename, iq))


def plot_residual(output_training, grad_color, base_color, color_incr, path, savename, legend=False, subselection=None):
    L = 1.0
    X_data = np.array(output_training.get('X_data'))
    u_data = np.array(output_training.get('u_data'))
    X_test = np.array(output_training.get('X_test'))
    u_test = np.array(output_training.get('u_test'))
    q_list = np.array(output_training.get('q'))
    n_alpha = output_training.get("n_alpha")[()]
    nq = len(q_list)

    alpha_reference = output_training.get('alpha_reference')[()]

    if subselection is None:
        alpha_list = range(n_alpha)
    else:
        alpha_list = subselection


    for iq, q in enumerate(q_list):
        plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
        ax = plt.subplot(1,1,1)
        for i_alpha in alpha_list:
            groupname = "alpha_%s" %(i_alpha)
            alpha = output_training.get(groupname + "/alpha")[()]
            u_pred = np.array(output_training.get(groupname + "/u_pred"))
            f_pred = np.array(output_training.get(groupname + "/f_pred"))

            if alpha >= np.finfo(float).eps: 
                significand, exponent = get_exponent(alpha/alpha_reference) 
                ax.semilogy(L - slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(np.abs(f_pred), iq, nq)[:,0], '-', linewidth=2,
                        color=grad_color(base_color + i_alpha * color_incr), label=r"$\alpha = %g \times 10^{%g} \bar{\alpha}$" %(significand, exponent))
                        # color=grad_color(base_color + i_alpha * color_incr), label=r"$\alpha = 10^{%g}$" %(log_alpha))
            else:
                ax.semilogy(L - slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(np.abs(f_pred), iq, nq)[:,0], '-', linewidth=2,
                        color=grad_color(base_color + i_alpha * color_incr), label=r"$\alpha = 0$")


        # powers = 1.0*(np.arange(0, 10)-8)
        # plt.yticks(10**powers)
        ax.set_xlabel(r"$x \; (\mathrm{m})$")
        ax.set_ylabel(r"$|f_{NN}|$")
        sq, eq = get_exponent(q)
        ax.set_title(r"$q = %.2f \times 10^{%d} \; (\mathrm{m}^2/\mathrm{s})$" %(sq, eq))
        ax.set_ylim([5*10**(-7), 5*10**2])

        # setting ticks in log scale
        y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 8)
        ax.yaxis.set_major_locator(y_major)
        y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.yaxis.set_minor_locator(y_minor)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        ax.grid(True, which="both")
        # ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
        if legend: 
            plt.legend()
        plt.tight_layout()

        if legend:
            plt.savefig(path + "figures/%s_residual_%d_legend.pdf" %(savename, iq))
        else:
            plt.savefig(path + "figures/%s_residual_%d.pdf" %(savename, iq))


# def plot_comparison(output_training, grad_color, base_color, color_incr, path, savename, subselection=None):
# 
#     if args.plot_comparison:
#         for i_alpha in range(n_alpha):
#             plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
#             groupname = "alpha_%s" %(i_alpha)
#             alpha = output_training.get(groupname + "/alpha")[()]
#             u_pred = np.array(output_training.get(groupname + "/u_pred"))
#             f_pred = np.array(output_training.get(groupname + "/f_pred"))
#         
#             plt.plot(u_test[:,0], u_pred[:,0], 'ob') 
#             if alpha > np.finfo(float).eps:
#                 log_alpha = np.log10(alpha)
#                 plt.title(r"$\alpha = 10^{%g}$" %(log_alpha))
#             else:
#                 plt.title(r"$\alpha = 0$")
#     
#             plt.xlabel("Test data")
#             plt.ylabel("Model prediction")
#     
#             plt.tight_layout()
#             plt.savefig(path + "figures/%s_comparison_%d.pdf" %(args.model, i_alpha))


def plot_lcurve(output_training, model, grad_color, base_color, color_incr, path, savename):
    plt.figure(figsize=FIGSIZE, dpi=FIGDPI)
    data_misfits = []
    pde_misfits = [] 
    L = 1.0
    X_data = np.array(output_training.get('X_data'))
    u_data = np.array(output_training.get('u_data'))
    q_list = np.array(output_training.get('q'))
    n_alpha = output_training.get("n_alpha")[()]
    nq = len(q_list)
    alpha_reference = output_training.get("alpha_reference")[()]

    text_pos_x = np.ones(n_alpha)
    text_pos_y = np.ones(n_alpha)

    if model == "dupuit":
        text_pos_x[n_alpha-4] = 0.98
        text_pos_y[n_alpha-4] = 0.70
        text_pos_x[n_alpha-3] = 1.01
    else:
        text_pos_y[n_alpha-3] = 0.75
        
 
    for i_alpha in range(1, n_alpha):
        groupname = "alpha_%s" %(i_alpha)
        alpha = output_training.get(groupname + "/alpha")[()]
        u_train = np.array(output_training.get(groupname + "/u_train"))
        f_train = np.array(output_training.get(groupname + "/f_train"))

        data_misfit = np.linalg.norm(u_train[:,0] - u_data[:,0])
        pde_misfit = np.linalg.norm(f_train[:,0]) 

        data_misfits.append(data_misfit)
        pde_misfits.append(pde_misfit)
        plt.loglog(data_misfit, pde_misfit, 'o', color=grad_color(base_color + i_alpha * color_incr))

        if alpha >= np.finfo(float).eps: 
            significand, exponent = get_exponent(alpha/alpha_reference)
            plt.text(text_pos_x[i_alpha]*data_misfit, text_pos_y[i_alpha]*pde_misfit, r"$10^{%g}\bar{\alpha}$" %(exponent))
        else:
            plt.text(data_misfit, pde_misfit, r"0")
    plt.loglog(data_misfits, pde_misfits, ':k')

    # plt.grid(True)
    plt.xlabel(r'Data misfit $\sum\|h_{NN} - h_{data}\|^2$')
    plt.ylabel(r'PDE Misfit $\sum\|f_{NN}\|^2$')
    # plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
    plt.tight_layout()
    plt.savefig(path + "figures/%s_lcurve.pdf" %(savename))
        

def main():
    args = parse_args()
    path = "synthetic/regularization/"
    os.makedirs(path + "figures", exist_ok=True)
    
    if args.name:
        filename = path + args.name + ".h5"
        print("Plotting ", args.name)
    else:
        flow_model = args.model
        filename = path + "data_" + flow_model + ".h5"
        print("Plotting ", flow_model, " results")
    
    L = 1.0
    output_training = h5py.File(filename, "r")
    n_alpha = output_training.get('n_alpha')[()]
    print("N ALPHA:", n_alpha)

    alpha_reference = output_training.get("alpha_reference")[()]
    print(alpha_reference)
    grad_color = plt.cm.get_cmap('turbo', 12)
    base_color = 0.0
    top_color = 0.8
    color_incr = (top_color - base_color)/n_alpha

    subselection = [0, 1, 3, 5, 7, 9, 11]
    
    if args.plot_prediction:
        plot_prediction(output_training, grad_color, base_color, color_incr, path, args.model, args.legend, subselection)
    if args.plot_residual:
        plot_residual(output_training, grad_color, base_color, color_incr, path, args.model, args.legend, subselection)
    if args.plot_lcurve:
        plot_lcurve(output_training, args.model, grad_color, base_color, color_incr, path, args.model)

    plt.show()

if __name__ == "__main__":
    main()

#         
