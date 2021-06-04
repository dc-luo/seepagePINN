import numpy as np 
import h5py 
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Select PDE model') 
parser.add_argument('-m', '--model', type=str, default="dinucci", choices=["dinucci", "dupuit"], help='PDE choice: dinucci, dupuit')
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


flow_model = args.model
filename = "data_" + flow_model + ".h5"

out_file = h5py.File(filename, "r")
qs = np.array(out_file.get('q_test'))
nq = qs.shape[0]

X_data = np.array(out_file.get('X_data'))
u_data = np.array(out_file.get('u_data'))
X_test = np.array(out_file.get('X_test'))
u_test = np.array(out_file.get('u_test'))

if args.plot_prediction:

    for iq, q in enumerate(qs):
        plt.figure()
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_test, iq, nq)[:,0], '--k', label="PDE")
        
        # No collocation points
        groupname = "no_colloc"
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-', label="No colloc")
    
        # With collocation points
        groupname = "colloc"
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], '-', label="Colloc")
    
        plt.title("q = %g" %(q))
        plt.legend()
    plt.show()

if args.plot_residual:
    for iq, q in enumerate(qs):
        plt.figure()
        # No collocation points
        groupname = "no_colloc"
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f_pred, iq, nq)[:,0], '-', label="No colloc")
    
        # With collocation points
        groupname = "colloc"
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        plt.plot(slice_to_flow(X_test, iq, nq)[:,0], slice_to_flow(f_pred, iq, nq)[:,0], '-', label="Colloc")
    
        plt.title("q = %g" %(q))
        plt.legend()
    
    plt.show()

if args.plot_comparison:
    plt.figure()
    q_train_min = 0.1
    q_train_max = 0.2

    for iq, q in enumerate(qs):
        # No collocation points
        groupname = "no_colloc"
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        if q <= q_train_max and q >= q_train_min:
            plt.plot(slice_to_flow(u_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], 'ob')
        else:
            plt.plot(slice_to_flow(u_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], 'ob')

    plt.xlabel("Test data")
    plt.ylabel("Prediction")
    plt.title("No collocation")
    plt.xlim([0, None])
    plt.ylim([0, None])

    plt.figure()

    for iq, q in enumerate(qs):
        # No collocation points
        groupname = "colloc"
        u_pred = np.array(out_file.get(groupname + "/u_pred"))
        f_pred = np.array(out_file.get(groupname + "/f_pred"))
        if q <= q_train_max and q >= q_train_min:
            plt.plot(slice_to_flow(u_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], 'ob')
        else:
            plt.plot(slice_to_flow(u_test, iq, nq)[:,0], slice_to_flow(u_pred, iq, nq)[:,0], 'ob')

    plt.xlabel("Test data")
    plt.ylabel("Prediction")
    plt.title("With collocation")
    plt.xlim([0, None])
    plt.ylim([0, None])

    plt.show()
