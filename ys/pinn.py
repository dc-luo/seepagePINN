import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time


class PhysicsInformedNN:
    """ class for physics informed nn """

    def __init__(self, X, X_left_boundary, X_right_boundary, u, layers, lb, ub, X_colloc=None, alpha=1, alpha_colloc = 1, betas=[0.,0.], optimizer_type="both"):
    
        self.lb = lb
        self.ub = ub

        self.x = X[:, 0:1]
        self.t = X[:, 1:2]
        self.u = u
        
        self.layers = layers
        self.optimizer_type = optimizer_type
        
        # left and right Neumann conditions
        self.alpha = alpha
        self.alpha_const = tf.constant(self.alpha, dtype=tf.float32, shape=[1,1])
        
#        self.b = b
#        self.b_const = tf.constant(self.b, dtype=tf.float32, shape=[1,1])

        self.beta_L = tf.constant(betas[0], dtype=tf.float32, shape=[1,1])
        self.beta_R = tf.constant(betas[1], dtype=tf.float32, shape=[1,1])
        
        #Collocation points
        self.X_colloc = X_colloc

        # initialise the NN
        # tf variables
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

#        # initialise parameters
#        self.lambda_1 = tf.Variable([np.log(0.05)], dtype=tf.float32) # k = e^{lambda_1}
#        # self.k = tf.constant([0.1], dtype=tf.float32)
#        # self.lambda_1 = tf.constant([np.log(k)], dtype=tf.float32)
#        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32) # neumann BC

        # define the placeholder variables
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        # define boundary variables
        self.x_left_boundary = X_left_boundary[:,0:1]
        self.x_right_boundary = X_right_boundary[:,0:1]
        self.t_boundary = X_left_boundary[:,1:2]
        
        # define boundary placeholders
        self.x_left_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.x_left_boundary.shape[1]])
        self.x_right_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.x_right_boundary.shape[1]])
        self.t_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.t_boundary.shape[1]])

        # NN structure for u and f
        self.u_pred = self.net_u(self.x_tf, self.t_tf)
        self.f_pred = self.net_f(self.x_tf, self.t_tf)

        # boundary residual
        self.f_left_boundary = self.net_left_boundary(self.x_left_boundary_tf, self.t_boundary_tf)
        self.f_right_boundary = self.net_right_boundary(self.x_right_boundary_tf, self.t_boundary_tf)
        
        if X_colloc is None:
        
            self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) \
                          + self.alpha_const * tf.reduce_mean(tf.square(self.f_pred)) \
                          + self.beta_L * tf.reduce_mean(tf.square(self.f_left_boundary)) \
                          + self.beta_R * tf.reduce_mean(tf.square(self.f_right_boundary))
        else:
            # Using collocation points X_colloc for PDE residual as well
            self.x_colloc = X_colloc[:, 0:1] # x points
            self.t_colloc = X_colloc[:, 1:2] # q points
            self.x_colloc_tf = tf.placeholder(tf.float32, shape=[None, self.x_colloc.shape[1]])
            self.t_colloc_tf = tf.placeholder(tf.float32, shape=[None, self.t_colloc.shape[1]])
            
            # PDE residual at collaction points
            self.f_colloc = self.net_f(self.x_colloc_tf, self.t_colloc_tf)
            
            self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) \
                          + self.alpha_const * tf.reduce_mean(tf.square(self.f_pred)) \
                          + self.alpha_colloc_const * tf.reduce_mean(tf.square(self.f_colloc)) \
                          + self.beta_L * tf.reduce_mean(tf.square(self.f_left_boundary)) \
                          + self.beta_R * tf.reduce_mean(tf.square(self.f_right_boundary))
            
        # define a BFGS optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
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

    def net_u(self, x, t):
        """ neural network structure for u """
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        NotImplementedError("Child class should implement predict")
    
    def net_left_boundary(self, x, t):
        """ left boundary residual """
        NotImplementedError("Child class should implement predict")

    def net_right_boundary(self, x, t):
        """ right boundary residual """
        NotImplementedError("Child class should implement predict")

    def train(self, nIter):
        NotImplementedError("Child class should implement predict")
        
    def predict(self, X_star):
        NotImplementedError("Child class should implement predict")

    def save(self, savename):
        """ saves model variables """
        saver = tf.train.Saver()
        savepath = saver.save(self.sess, "saved_models/%s" %(savename))
        print("Model saved in path: %s" %(savepath))

    def load(self, savename):
        saver = tf.train.Saver()
        saver.restore(self.sess, "saved_models/%s" %(savename))


##############################################################################
class DupuitPINN(PhysicsInformedNN):

    def __init__(self, X, X_left_boundary, X_right_boundary, u, lambda_1, lambda_2, layers, lb, ub, X_colloc=None, b=0, alpha=1, alpha_colloc=1, betas = [0.,0.], optimizer_type="adam"):

        # initialise parameters
        self.lambda_1 = tf.Variable([np.log(lambda_1)], dtype=tf.float32) # k = e^{lambda_1}
        self.lambda_2 = tf.Variable([lambda_2], dtype=tf.float32) # neumann BC

        self.b = b
        self.b_const = tf.constant(self.b, dtype=tf.float32, shape=[1,1])
        
        super().__init__(X, X_left_boundary, X_right_boundary, u, layers, lb, ub,
                X_colloc=X_colloc, alpha=alpha, alpha_colloc=alpha_colloc,
                betas=betas, optimizer_type=optimizer_type)

    
    def net_f(self, x, t):
        lambda_con1 = tf.exp(self.lambda_1)
        #source = tf.constant(self.q, dtype=tf.float32, shape=[1,1])
        u = self.net_u(x,t)
        u_x = tf.gradients(u, x)[0]
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        #f = lambda_1 * u_x * u + source
        f = u_t - lambda_con1 * (u_x * u_x + u * u_xx)
        # f = u_t - self.k * (u_x * u_x + u * u_xx)
        return f
    
    def net_left_boundary(self, x, t):
        """ left boundary residual """
        u = self.net_u(x,t)
        u_x = tf.gradients(u, x)[0]

        f = u_x - self.lambda_2
        return f

    def net_right_boundary(self, x, t):
        """ right boundary residual """
        u = self.net_u(x,t)
        u_x = tf.gradients(u, x)[0]

        f = u_x - self.b_const
        return f

    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, np.exp(lambda_1), lambda_2))

    def train(self, nIter):
        
        if self.X_colloc is None:
            tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u,
                    self.x_left_boundary_tf : self.x_left_boundary,
                    self.x_right_boundary_tf : self.x_right_boundary,
                    self.t_boundary_tf : self.t_boundary}
        else:
            tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u,
                    self.x_left_boundary_tf : self.x_left_boundary,
                    self.x_right_boundary_tf : self.x_right_boundary,
                    self.t_boundary_tf : self.t_boundary,
                    self.x_colloc_tf:self.x_colloc,
                    self.t_colloc_tf:self.t_colloc}
                        
                        
        start_time = time.time()
        if self.optimizer_type == "adam":
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)

                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    lambda_2_value = self.sess.run(self.lambda_2)

                    print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.3f, Time: %.2f' %
                          (it, loss_value, np.exp(lambda_1_value), lambda_2_value, elapsed))
                    start_time = time.time()

            self.optimizer.minimize(self.sess,
                                     feed_dict = tf_dict,
                                     fetches = [self.loss, self.lambda_1, self.lambda_2],
                                     loss_callback = self.callback)

        else:
            self.optimizer.minimize(self.sess,
                                     feed_dict = tf_dict,
                                     fetches = [self.loss, self.lambda_1, self.lambda_2],
                                     loss_callback = self.callback)
        
    def predict(self, X_star):
        tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2],
                self.x_left_boundary_tf : self.x_left_boundary,
                self.x_right_boundary_tf : self.x_right_boundary,
                self.t_boundary_tf : self.t_boundary}

        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        f_left = self.sess.run(self.f_left_boundary, tf_dict)
        f_right = self.sess.run(self.f_right_boundary, tf_dict)

        return u_star, f_star, f_left, f_right

