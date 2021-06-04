import numpy as np 
import h5py 
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Select PDE model') 
parser.add_argument('-m', '--model', type=int, default=0, choices=[0,1], help='PDE choice: 0 = dinucci, 1 = dupuit')
parser.add_argument("-t", "--plot_trends", help="plot model vs prediction as trends", action="store_true", default=False)
parser.add_argument("-c", "--plot_comparison", help="plot model vs prediction as scatter", action="store_true", default=False)
parser.add_argument("-l", "--plot_lcurve", help="plot L-curve analysis", action="store_true", default=False)
args = parser.parse_args()


def slice_to_flow(arr, i, n):
    """ slice to the ith flow value given a total of n possible flow values""" 
    assert(arr.shape[0] % n == 0)
    incr = round(arr.shape[0]/n)
    i_lower = i * incr
    i_upper = (i+1) * incr
    return arr[i_lower:i_upper, :]

if args.model == 0:
    flow_model = "dinucci"
elif args.model == 1:
    flow_model = "dupuit"

print("Plotting ", flow_model, " results")

filename = "data_" + flow_model + ".h5"

output_training = h5py.File(filename, "r")
qs = np.array(output_training.get('q'))
nq = qs.shape[0]
n_alpha = output_training.get('n_alpha')[()]

X_data = np.array(output_training.get('X_data'))
u_data = np.array(output_training.get('u_data'))
X_test = np.array(output_training.get('X_test'))
u_test = np.array(output_training.get('u_test'))

if args.plot_trends:
    for iq, q in enumerate(qs):
        plt.figure()
        plt.plot(slice_to_flow(X_data, iq, nq)[:,0], slice_to_flow(u_data, iq, nq)[:,0], 'ok', label="Data")
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_test, iq, nq)[:,0], '--k', label="PDE")
    
        for i_alpha in range(n_alpha):
            groupname = "alpha_%s" %(i_alpha)
            alpha = output_training.get(groupname + "/alpha")[()]
            u_pred = np.array(output_training.get(groupname + "/u_pred"))
            f_pred = np.array(output_training.get(groupname + "/f_pred"))
    
            plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-', label="a = %g" %(alpha))
    
        plt.title("q = %g" %(q))
        plt.legend()
    
    plt.show()

if args.plot_comparison:
    for i_alpha in range(n_alpha):
        plt.figure()
        groupname = "alpha_%s" %(i_alpha)
        alpha = output_training.get(groupname + "/alpha")[()]
        u_pred = np.array(output_training.get(groupname + "/u_pred"))
        f_pred = np.array(output_training.get(groupname + "/f_pred"))
    
        plt.plot(u_test[:,0], u_pred[:,0], 'o') 
        plt.title("alpha = %g" %(alpha))
    
    plt.show()
    
if args.plot_lcurve:
    plt.figure()
    for i_alpha in range(n_alpha):
        groupname = "alpha_%s" %(i_alpha)
        alpha = output_training.get(groupname + "/alpha")[()]
        u_pred = np.array(output_training.get(groupname + "/u_pred"))
        f_pred = np.array(output_training.get(groupname + "/f_pred"))

        data_misfit = np.linalg.norm(u_pred[:,0] - u_test[:,0])
        pde_misfit = np.linalg.norm(f_pred[:,0]) 

        plt.loglog(data_misfit, pde_misfit, 'ob')
    plt.show()
