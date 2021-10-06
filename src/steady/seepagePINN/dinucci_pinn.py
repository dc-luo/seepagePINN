import tensorflow as tf 
import numpy as np 
import time 
from .pinn2d import PINN2D

class DiNucciNormalizedScaledPINNFlow(PINN2D):
    """ 
    PINNS model for Di-Nucci flow equation, where the NN takes the input
    import time 
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

class DiNucciNormalizedScaledPINNFitAll(PINN2D):
    """ 
    PINNS model for Di-Nucci flow equation, where the NN takes the input
    x-location and input flow q 

    The flow equation is normalized by dividing by q such that the source term is O(1)

    input data q is scaled as q_input = q_raw/scale_q
    this is rescaled to q_raw = q * scale_q within the residual
    """

    def __init__(self, X, u, kappa, layers, lb, ub, scale_q=1.0, X_colloc=None, alpha=1, alpha_colloc=1,
            params_positive = False, optimizer_type="adam"):
        self.kappa = kappa
        self.scale_q = scale_q
        self.params_positive = params_positive

        # optimization variable for K and coefficients for dinucci terms 
        self.lambda_1 = tf.Variable([np.log(self.kappa)], dtype=tf.float32)
        if self.params_positive:
            # use default values of np.log(1) = 0 
            self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
            self.lambda_3 = tf.Variable([0.0], dtype=tf.float32)
        else:
            # use default values of 1
            self.lambda_2 = tf.Variable([1.0], dtype=tf.float32)
            self.lambda_3 = tf.Variable([1.0], dtype=tf.float32)

        super().__init__(X, u, layers, lb, ub, X_colloc=X_colloc, alpha=alpha, alpha_colloc=alpha_colloc, optimizer_type=optimizer_type)

        self.f1_pred = self.net_f1(self.x_tf, self.q_tf)
        self.f2_pred = self.net_f2(self.x_tf, self.q_tf)
        self.f3_pred = self.net_f3(self.x_tf, self.q_tf)

    def net_f(self, x, q):
        kappa = tf.exp(self.lambda_1)
        if self.params_positive:
            # Transform by exp to make parameter positive 
            term2 = tf.exp(self.lambda_2)
            term3 = tf.exp(self.lambda_3) 
        else:
            term2 = self.lambda_2
            term3 = self.lambda_3

        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        f = 1 + kappa/(q*self.scale_q)*u*u_x + term2/3*u*u_xx + term3/3*u_x*u_x 
        return f
        
    def callback(self, loss, lambda_1, lambda_2, lambda_3):
        if self.params_positive:
            print('Loss: %e, K: %.5g, l2: %.5g, l3: %.5g' %(loss, 
                np.exp(lambda_1), 
                np.exp(lambda_2),
                np.exp(lambda_3))
                ) 
        else:
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
                    if self.params_positive:
                        print('It: %d, Loss: %.3e, K: %.3f, Lambda_2: %.3f, Lambda_3: %.3f, Time: %.2f' % 
                          (it, loss_value, np.exp(lambda_1_value), 
                              np.exp(lambda_2_value), np.exp(lambda_3_value), elapsed))
                    else: 
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
                    if self.params_positive:
                        print('It: %d, Loss: %.3e, K: %.3f, Lambda_2: %.3f, Lambda_3: %.3f, Time: %.2f' % 
                          (it, loss_value, np.exp(lambda_1_value), 
                              np.exp(lambda_2_value), np.exp(lambda_3_value), elapsed))
                    else: 
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


    def net_f1(self, x, q):

        kappa = tf.exp(self.lambda_1)
        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        f = 1/(q*self.scale_q)*u*u_x 

        return f


    def net_f2(self, x, q):
        
        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = 1/3*u*u_xx 

        return f


    def net_f3(self, x, q):

        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = 1/3*u_x*u_x 

        return f

    def predict_terms(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.q_tf: X_star[:,1:2]}
        
        f1_star = self.sess.run(self.f1_pred, tf_dict)
        f2_star = self.sess.run(self.f2_pred, tf_dict)
        f3_star = self.sess.run(self.f3_pred, tf_dict)
        
        return f1_star, f2_star, f3_star
    
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

        # For predicting the individual terms
        self.f1_pred = self.net_f1(self.x_tf, self.q_tf)
        self.f2_pred = self.net_f2(self.x_tf, self.q_tf)
        self.f3_pred = self.net_f3(self.x_tf, self.q_tf)

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

    def net_f1(self, x, q):

        kappa = tf.exp(self.lambda_1)
        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        f = 1/(q*self.scale_q)*u*u_x 

        return f


    def net_f2(self, x, q):
        
        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = 1/3*u*u_xx 

        return f


    def net_f3(self, x, q):

        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = 1/3*u_x*u_x 

        return f

    def predict_terms(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.q_tf: X_star[:,1:2]}
        
        f1_star = self.sess.run(self.f1_pred, tf_dict)
        f2_star = self.sess.run(self.f2_pred, tf_dict)
        f3_star = self.sess.run(self.f3_pred, tf_dict)
        
        return f1_star, f2_star, f3_star
