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

def slice_to_flow(arr, i, n):
    """ slice to the ith flow value given a total of n possible flow values""" 
    assert(arr.shape[0] % n == 0)
    incr = round(arr.shape[0]/n)
    i_lower = i * incr
    i_upper = (i+1) * incr
    return arr[i_lower:i_upper, :]

data_model = args.data_model
flow_model = args.flow_model
filename = "data_" + data_model + "_" + flow_model + ".h5"

print(filename)

out_file = h5py.File(filename, "r")
qs = np.array(out_file.get('q'))
nq = qs.shape[0]

X_data = np.array(out_file.get('X_data'))
u_data = np.array(out_file.get('u_data'))
X_test = np.array(out_file.get('X_test'))
u_test = np.array(out_file.get('u_test'))

K_truth = out_file.get('K_truth')[()]

if args.plot_prediction:

    for iq, q in enumerate(qs):
        plt.figure()
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_test, iq, nq)[:,0], 'ok', label="PDE")
        
        # No collocation points
        groupname = "alpha_large"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-', label="a = %g" %(alpha))
    
        # With collocation points
        groupname = "alpha_small"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-', label="a = %g" %(alpha))
    
        plt.title("q = %g" %(q))
        plt.legend()
    plt.show()

if args.plot_residual:
    for iq, q in enumerate(qs):
        plt.figure()
        # No collocation points
        groupname = "alpha_large"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f_pred, iq, nq)[:,0], '-', label="a = %g" %(alpha))
    
        # With collocation points
        groupname = "alpha_small"
        alpha = out_file.get(groupname + "/alpha")[()]
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f_pred, iq, nq)[:,0], '-', label="a = %g" %(alpha))
    
        plt.title("q = %g" %(q))
        plt.legend()
    
    plt.show()

if args.plot_comparison:
    plt.figure()

    groupname = "alpha_large"
    alpha = out_file.get(groupname + "/alpha")[()]
    u_pred = np.array(out_file.get(groupname + "/u_pred"))
    f_pred = np.array(out_file.get(groupname + "/f_pred"))
    plt.plot(u_test[:,0], u_pred[:,0], 'ob')

    plt.xlabel("Test data")
    plt.ylabel("Prediction")
    plt.title("a = %g" %(alpha))
    plt.xlim([0, None])
    plt.ylim([0, None])

    plt.figure()

    groupname = "alpha_small"
    alpha = out_file.get(groupname + "/alpha")[()]
    u_pred = np.array(out_file.get(groupname + "/u_pred"))
    f_pred = np.array(out_file.get(groupname + "/f_pred"))
    plt.plot(u_test[:,0], u_pred[:,0], 'ob')

    plt.xlabel("Test data")
    plt.ylabel("Prediction")
    plt.title("a = %g" %(alpha))
    plt.xlim([0, None])
    plt.ylim([0, None])

    plt.show()
