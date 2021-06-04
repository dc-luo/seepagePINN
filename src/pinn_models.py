import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time


class PINN2D:
    """ Base class for a physics informed neural network """

    def __init__(self, X, u, layers, lb, ub, X_colloc=None, alpha=1, alpha_colloc=1, optimizer_type="both"):
        self.lb = lb # lower bound for x 
        self.ub = ub # upper bound for x
        self.layers = layers # architecture 
        self.alpha = alpha  # regularization parameter
        self.alpha_colloc = alpha_colloc  # regularization parameter
        self.X_colloc = X_colloc # Collocation points
        self.optimizer_type = optimizer_type # Optimizer type

        # Define the regularization parameter as tf.constant
        self.alpha_const = tf.constant(self.alpha, dtype=tf.float32, shape=[1,1])
        self.alpha_colloc_const = tf.constant(self.alpha_colloc, dtype=tf.float32, shape=[1,1])

        self.x = X[:,0:1]
        self.q = X[:,1:2]
        self.u = u
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graphs
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        # inputs and outputs
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.q_tf = tf.placeholder(tf.float32, shape=[None, self.q.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.u_pred = self.net_u(self.x_tf, self.q_tf)
        self.f_pred = self.net_f(self.x_tf, self.q_tf)
        
        if X_colloc is None:
            # Data misfit + PDE misfit at X 
            self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                        self.alpha_const * tf.reduce_mean(tf.square(self.f_pred))
        else:
            # Using collocation points X_colloc for PDE residual as well  
            self.x_colloc = X_colloc[:, 0:1] # x points
            self.q_colloc = X_colloc[:, 1:2] # q points 
            self.x_colloc_tf = tf.placeholder(tf.float32, shape=[None, self.x_colloc.shape[1]]) 
            self.q_colloc_tf = tf.placeholder(tf.float32, shape=[None, self.q_colloc.shape[1]])

            # PDE residual at collaction points 
            self.f_colloc = self.net_f(self.x_colloc_tf, self.q_colloc_tf) 

            # Loss to account for training data on the state, PDE residual of data, and PDE resdiaul
            # at collocation points
            self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) \
                        + self.alpha_const * tf.reduce_mean(tf.square(self.f_pred)) \
                        + self.alpha_colloc_const * tf.reduce_mean(tf.square(self.f_colloc))
        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 80,
                                                                           'maxls': 80,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        """ 
        takes input variable `layers`, a list of layer widths
        initialise the nn and return weights, biases
        lists of tf.Variables corresponding to each layer 
        """
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases

    def xavier_init(self, size):
        """ Xavier initialization for the weights """ 
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        """ returns the graph for the nn """

        num_layers = len(weights) + 1 

        # Rescale the variable X 
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1 

        # Apply sigma(WX + b) to each layer up until last one
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b)) 

        # last layer, no activation function 
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) 
        return Y

    def net_u(self, x, q):  
        u = self.neural_net(tf.concat([x,q],1), self.weights, self.biases)
        return u

    def net_f(self, x, q):
        raise NotImplementedError("Child class should implement net_f")
    
    def save(self, savename):
        """ saves model variables """
        saver = tf.train.Saver()
        savepath = saver.save(self.sess, "saved_models/%s" %(savename))
        print("Model saved in path: %s" %(savepath))

    def load(self, savename):
        saver = tf.train.Saver()
        saver.restore(self.sess, "saved_models/%s" %(savename))

    def train(self, nIter):
        raise NotImplementedError("Child class should implement train")
    
    def predict(self, X_data):
        raise NotImplementedError("Child class should implement predict")


class DupuitNormalizedScaledPINNFitK(PINN2D):
    # Initialize the class
    def __init__(self, X, u, kappa, layers, lb, ub, scale_q=1.0, X_colloc=None, alpha=1, alpha_colloc=1, optimizer_type="adam"):
        """ 
        PINNS model for Dupuit appxorimation flow equation, where the NN takes the input
        x-location and input flow q 

        The flow equation is normalized by dividing by q such that the source term is O(1)

        input data q is scaled as q_input = q_raw/scale_q
        this is rescaled to q_raw = q * scale_q within the residual
        """

        # Initialize with PINN for weights, biases, sess
        self.kappa = kappa # default value for kappa
        self.lambda_1 = tf.Variable([np.log(self.kappa)], dtype=tf.float32)
        self.scale_q = scale_q # scaling of data q

        super().__init__(X, u, layers, lb, ub, X_colloc=X_colloc, alpha=alpha, alpha_colloc=alpha_colloc, optimizer_type=optimizer_type)
        
    def net_f(self, x, q):
        kappa = tf.exp(self.lambda_1)
        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        f = 1 + kappa/(q*self.scale_q)*u*u_x
        return f
    
    def callback(self, loss, lambda_1):
        print('Loss: %e, K: %.5g' %(loss, np.exp(lambda_1))) 

    def train(self, nIter):
        if self.X_colloc is None:
            tf_dict = {self.x_tf: self.x, self.q_tf: self.q, self.u_tf: self.u}
               
        else: 
            tf_dict = {self.x_tf: self.x, self.q_tf: self.q, self.u_tf: self.u, 
                    self.x_colloc_tf: self.x_colloc, self.q_colloc_tf: self.q_colloc}
        
        start_time = time.time()
        if self.optimizer_type == "adam":
            # use adam only
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    print('It: %d, Loss: %.3e, K: %.3f, Time: %.2f' % 
                          (it, loss_value, np.exp(lambda_1_value), elapsed))
                    start_time = time.time()

        elif self.optimizer_type == "both":
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    print('It: %d, Loss: %.3e, K: %.3f, Time: %.2f' % 
                          (it, loss_value, np.exp(lambda_1_value), elapsed))
                    start_time = time.time()

            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss, self.lambda_1],
                                    loss_callback = self.callback)

        else: 
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss, self.lambda_1],
                                    loss_callback = self.callback)
        
        
    def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.q_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star, f_star



class DiNucciNormalizedScaledPINNFitK(PINN2D):
    """ 
    PINNS model for Di-Nucci flow equation, where the NN takes the input
    x-location and input flow q 

    The flow equation is normalized by dividing by q such that the source term is O(1)

    input data q is scaled as q_input = q_raw/scale_q
    this is rescaled to q_raw = q * scale_q within the residual
    """

    def __init__(self, X, u, kappa, layers, lb, ub, scale_q=1.0, X_colloc=None, alpha=1, alpha_colloc=1, optimizer_type="adam"):
        self.kappa = kappa
        self.scale_q = scale_q
        # optimization variable for K 
        self.lambda_1 = tf.Variable([np.log(self.kappa)], dtype=tf.float32)

        super().__init__(X, u, layers, lb, ub, X_colloc=X_colloc, alpha=alpha, alpha_colloc=alpha_colloc, optimizer_type=optimizer_type)

    def net_f(self, x, q):
        kappa = tf.exp(self.lambda_1)
        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        f = 1 + kappa/(q*self.scale_q)*u*u_x + 1/3*u*u_xx + 1/3*u_x*u_x 
        return f
    
    def callback(self, loss, lambda_1):
        print('Loss: %e, K: %.3f' %(loss, np.exp(lambda_1))) 
        
    def train(self, nIter):
        if self.X_colloc is None:
            tf_dict = {self.x_tf: self.x, self.q_tf: self.q, self.u_tf: self.u}
               
        else: 
            tf_dict = {self.x_tf: self.x, self.q_tf: self.q, self.u_tf: self.u, 
                    self.x_colloc_tf: self.x_colloc, self.q_colloc_tf: self.q_colloc}
        
        start_time = time.time()
        if self.optimizer_type == "adam":
            # use adam only
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    print('It: %d, Loss: %.3e, K: %.3f, Time: %.2f' % 
                          (it, loss_value, np.exp(lambda_1_value), elapsed))
                    start_time = time.time()

        elif self.optimizer_type == "both":
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    print('It: %d, Loss: %.3e, K: %.3f, Time: %.2f' 
                            %(it, loss_value, np.exp(lambda_1_value), elapsed))
                    start_time = time.time()

            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss, self.lambda_1],
                                    loss_callback = self.callback)

        else: 
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss, self.lambda_1],
                                    loss_callback = self.callback)
        
        
    def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.q_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star, f_star
        
class DiNucciNormalizedScaledPINNFitAll(PINN2D):
    """ 
    PINNS model for Di-Nucci flow equation, where the NN takes the input
    x-location and input flow q 

    The flow equation is normalized by dividing by q such that the source term is O(1)

    input data q is scaled as q_input = q_raw/scale_q
    this is rescaled to q_raw = q * scale_q within the residual
    """

    def __init__(self, X, u, kappa, layers, lb, ub, scale_q=1.0, X_colloc=None, alpha=1, alpha_colloc=1, optimizer_type="adam"):
        self.kappa = kappa
        self.scale_q = scale_q

        # optimization variable for K and coefficients for dinucci terms 
        self.lambda_1 = tf.Variable([np.log(self.kappa)], dtype=tf.float32)
        self.lambda_2 = tf.Variable([1.0], dtype=tf.float32)
        self.lambda_3 = tf.Variable([1.0], dtype=tf.float32)

        super().__init__(X, u, layers, lb, ub, X_colloc=X_colloc, alpha=alpha, alpha_colloc=alpha_colloc, optimizer_type=optimizer_type)

    def net_f(self, x, q):
        kappa = tf.exp(self.lambda_1)
        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        f = 1 + kappa/(q*self.scale_q)*u*u_x + self.lambda_2/3*u*u_xx + self.lambda_2/3*u_x*u_x 
        return f
        
    def callback(self, loss, lambda_1, lambda_2, lambda_3):
        print('Loss: %e, K: %.5g, l2: %.5g, l3: %.5g' %(loss, np.exp(lambda_1), lambda_2, lambda_3)) 
        
    def train(self, nIter):
        if self.X_colloc is None:
            tf_dict = {self.x_tf: self.x, self.q_tf: self.q, self.u_tf: self.u}
               
        else: 
            tf_dict = {self.x_tf: self.x, self.q_tf: self.q, self.u_tf: self.u, 
                    self.x_colloc_tf: self.x_colloc, self.q_colloc_tf: self.q_colloc}
        
        start_time = time.time()
        if self.optimizer_type == "adam":
            # use adam only
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    lambda_2_value = self.sess.run(self.lambda_2)
                    lambda_3_value = self.sess.run(self.lambda_3)
                    print('It: %d, Loss: %.3e, K: %.3f, Lambda_2: %.3f, Lambda_3: %.3f, Time: %.2f' % 
                          (it, loss_value, np.exp(lambda_1_value), lambda_2_value, lambda_3_value, elapsed))
                    start_time = time.time()

        elif self.optimizer_type == "both":
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    lambda_2_value = self.sess.run(self.lambda_2)
                    lambda_3_value = self.sess.run(self.lambda_3)
                    print('It: %d, Loss: %.3e, K: %.3f, Lambda_2: %.3f, Lambda_3: %.3f, Time: %.2f' % 
                          (it, loss_value, np.exp(lambda_1_value), lambda_2_value, lambda_3_value, elapsed))
                    start_time = time.time()

            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss, self.lambda_1, self.lambda_2, self.lambda_3],
                                    loss_callback = self.callback)

        else: 
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss, self.lambda_1, self.lambda_2, self.lambda_3],
                                    loss_callback = self.callback)
        
        
    def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.q_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star, f_star

class DupuitNormalizedScaledPINNFlow(PINN2D):
    """ 
    PINNS model for Dupuit flow equation, where the NN takes the input
    x-location and input flow q 

    The flow equation is normalized by dividing by q such that the source term is O(1)

    input data q is scaled as q_input = q_raw/scale_q
    this is rescaled to q_raw = q * scale_q within the residual
    """

    def __init__(self, X, u, kappa, layers, lb, ub, scale_q=1.0, X_colloc=None, alpha=1, alpha_colloc=1, optimizer_type="adam"):
        self.kappa = kappa # permeability
        self.scale_q = scale_q # scaling of q data 
        super().__init__(X, u, layers, lb, ub, X_colloc=X_colloc, alpha=alpha, alpha_colloc=alpha_colloc, optimizer_type=optimizer_type)

    def net_f(self, x, q):
        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        f = 1 + self.kappa/(q*self.scale_q)*u*u_x 
        return f
    
    def callback(self, loss):
        print('Loss: %e' %(loss)) 
        
    def train(self, nIter):
        if self.X_colloc is None:
            tf_dict = {self.x_tf: self.x, self.q_tf: self.q, self.u_tf: self.u}
               
        else: 
            tf_dict = {self.x_tf: self.x, self.q_tf: self.q, self.u_tf: self.u, 
                    self.x_colloc_tf: self.x_colloc, self.q_colloc_tf: self.q_colloc}
        
        start_time = time.time()
        if self.optimizer_type == "adam":
            # use adam only
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.3e, Time: %.2f' % 
                          (it, loss_value, elapsed))
                    start_time = time.time()

        elif self.optimizer_type == "both":
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.3e, Time: %.2f' % 
                          (it, loss_value, elapsed))
                    start_time = time.time()

            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback)

        else:
            # use BFGS 
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback)
        
        
    def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.q_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star, f_star


class DiNucciNormalizedScaledPINNFlow(PINN2D):
    """ 
    PINNS model for Di-Nucci flow equation, where the NN takes the input
    x-location and input flow q 

    The flow equation is normalized by dividing by q such that the source term is O(1)

    input data q is scaled as q_input = q_raw/scale_q
    this is rescaled to q_raw = q * scale_q within the residual
    """

    def __init__(self, X, u, kappa, layers, lb, ub, scale_q=1.0, X_colloc=None, alpha=1, alpha_colloc=1, optimizer_type="adam"):
        self.kappa = kappa # permeability
        self.scale_q = scale_q # scaling of q data 
        super().__init__(X, u, layers, lb, ub, X_colloc=X_colloc, alpha=alpha, alpha_colloc=alpha_colloc, optimizer_type=optimizer_type)

    def net_f(self, x, q):
        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        f = 1 + self.kappa/(q*self.scale_q)*u*u_x + 1/3*u*u_xx + 1/3*u_x*u_x 
        return f
    
    def callback(self, loss):
        print('Loss: %e' %(loss)) 
        
    def train(self, nIter):
        if self.X_colloc is None:
            tf_dict = {self.x_tf: self.x, self.q_tf: self.q, self.u_tf: self.u}
               
        else: 
            tf_dict = {self.x_tf: self.x, self.q_tf: self.q, self.u_tf: self.u, 
                    self.x_colloc_tf: self.x_colloc, self.q_colloc_tf: self.q_colloc}
        
        start_time = time.time()
        if self.optimizer_type == "adam":
            # use adam only
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.3e, Time: %.2f' % 
                          (it, loss_value, elapsed))
                    start_time = time.time()

        elif self.optimizer_type == "both":
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.3e, Time: %.2f' % 
                          (it, loss_value, elapsed))
                    start_time = time.time()

            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback)

        else:
            # use BFGS 
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback)
        
        
    def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.q_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star, f_star


