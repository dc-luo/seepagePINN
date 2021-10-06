import tensorflow as tf
import numpy as np 
import time 
from .pinn2d import PINN2D


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


