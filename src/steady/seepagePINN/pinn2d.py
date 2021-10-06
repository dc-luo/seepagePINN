import tensorflow as tf
import numpy as np

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
        savepath = saver.save(self.sess, "%s" %(savename))
        print("Model saved in path: %s" %(savepath))

    def load(self, savename):
        saver = tf.train.Saver()
        saver.restore(self.sess, "%s" %(savename))

    def train(self, nIter):
        raise NotImplementedError("Child class should implement train")
    
    def predict(self, X_data):
        raise NotImplementedError("Child class should implement predict")


