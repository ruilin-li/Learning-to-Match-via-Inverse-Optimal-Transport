import numpy as np
import scipy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import polynomial_kernel
from scipy.optimize import minimize
import ot

class train_parameters(object):
    def __init__(self, max_outer_iteration, max_inner_iteration, learning_rate):
        self.max_outer_iteration = max_outer_iteration
        self.max_inner_iteration = max_inner_iteration
        self.learning_rate = learning_rate


class model_parameters(object):
    def __init__(self, A0, gamma, const, degree, lam, lambda_mu, lambda_nu, delta):
        self.A0 = A0
        self.gamma = gamma 
        self.const = const
        self.degree = degree
        self.lam = lam
        self.lambda_mu = lambda_mu
        self.lambda_nu = lambda_nu
        self.delta = delta


class Matcher():
    def __init__(self, pi_sample, U0, V0, r=5):
        self.pi_sample = pi_sample
        self.U0 = U0
        self.V0 = V0
        self.p, self.m = U0.shape
        self.q, self.n = V0.shape
        self.r = r


    def polynomial_kernel(self, model_param, train_param, method='gd'):
        # unpack model parameters
        A0 = model_param.A0
        gamma = model_param.gamma 
        const = model_param.const
        degree = model_param.degree
        lam = model_param.lam
        lambda_mu = model_param.lambda_mu
        lambda_nu = model_param.lambda_nu
        delta = model_param.delta

        # gamma, const, degree = kernel_param[0], 1, kernel_param[1]
        # lam = lambda_nu = lambda_mu = 1

        # unpack training parameters
        max_iteration = train_param.max_outer_iteration
        learning_rate = train_param.learning_rate
        # max_iteration, learning_rate = train_param[0], train_param[1]

        A = A0
        r = self.pi_sample.sum(axis=1)
        c = self.pi_sample.sum(axis=0)

        history = [ ]
        keys = ['iteration', 'loss', 'diff_pi_sample']

        if method == 'gd':        
            for i in range(max_iteration):
                # update A
                C = np.power(gamma * self.U0.T.dot(A).dot(self.V0) + const, degree)
                pi = ot.rot(C, r, c)[0]
                factor = degree * gamma * np.power(gamma * self.U0.T.dot(A).dot(self.V0) + const, degree - 1)
                M = (self.pi_sample - pi) * factor
                grad_A = self.U0.dot(M).dot(self.V0.T)
                A -= learning_rate * grad_A

                # compute loss
                loss = np.sum(self.pi_sample * C)  \
                        - np.sum(pi * C)      \
                        - np.sum(pi * np.log(pi))

                values = [i+1, loss, np.linalg.norm(pi - self.pi_sample)]
                history.append(dict(zip(keys, values)))

        if method == 'bfgs':
            iteration = [0]

            def func(A):
                A = A.reshape(self.p, self.q)
                C = np.power(gamma * self.U0.T.dot(A).dot(self.V0) + 1, degree)
                pi = ot.rot(C, r, c)[0]
                loss = np.sum(self.pi_sample * C)  \
                        - np.sum(pi * C)      \
                        - np.sum(pi * np.log(pi))
                return loss 

            def grad(A):
                A = A.reshape(self.p, self.q)
                C = np.power(gamma * self.U0.T.dot(A).dot(self.V0) + 1, degree)
                pi = ot.rot(C, r, c)[0]
                factor = degree * gamma * np.power(gamma * self.U0.T.dot(A).dot(self.V0) + 1, degree - 1)
                M = (self.pi_sample - pi) * factor
                return self.U0.dot(M).dot(self.V0.T).ravel()

            def callback(A):
                A = A.reshape(self.p, self.q)
                C = np.power(gamma * self.U0.T.dot(A).dot(self.V0) + 1, degree)
                pi = ot.rot(C, r, c)[0]
                values = [iteration[0]+1, func(A),
                    np.linalg.norm(pi - self.pi_sample)]
                history.append(dict(zip(keys, values)))
                iteration[0] += 1
                          
            res = minimize(func, A.ravel(), method='BFGS', jac=grad, callback=callback, options={'gtol': 1e-8, 'disp': True})
            A = res.x.reshape(self.p, self.q)

        C = np.power(gamma * self.U0.T.dot(A).dot(self.V0) + 1, degree)
        pi = ot.rot(C, r, c)[0]
        return C, A, pi, history





    def riot(self, model_param, train_param):
        # unpack model parameters
        A0 = model_param.A0
        gamma = model_param.gamma 
        const = model_param.const
        degree = model_param.degree
        lam = model_param.lam
        lambda_mu = model_param.lambda_mu
        lambda_nu = model_param.lambda_nu
        delta = model_param.delta

        # unpack training parameters
        max_outer_iteration = train_param.max_outer_iteration
        max_inner_iteration = train_param.max_inner_iteration  
        learning_rate = train_param.learning_rate

        C1 = 5 * pairwise_distances(np.random.randn(self.m, 2))
        C2 = 5 * pairwise_distances(np.random.randn(self.n, 2))

        A = A0
        C = np.power(gamma * self.U0.T.dot(A).dot(self.V0) + const, degree)
        
        r_sample, c_sample = self.pi_sample.sum(axis=1), self.pi_sample.sum(axis=0)
        v = np.log(ot.rot(C1, r_sample, r_sample, lam=lambda_mu)[2]) / lambda_mu
        w = np.log(ot.rot(C2, c_sample, c_sample, lam=lambda_nu)[2]) / lambda_nu
        v_dual = (np.log(r_sample) - np.log(np.sum(np.exp(lambda_mu * (np.outer(np.ones(self.m), v) - C1)), axis=0))) / lambda_mu
        w_dual = (np.log(c_sample) - np.log(np.sum(np.exp(lambda_nu * (np.outer(np.ones(self.n), w) - C2)), axis=0))) / lambda_nu
        
        pi, xi, eta = ot.rot(C, r_sample, c_sample, lam)[:-1]
        
        def KL(pi1, pi2):
            p, q = pi1.ravel(), pi2.ravel()
            return np.sum(p * np.log(p / q))
        
        def rel_error(M, M0):
            return np.linalg.norm(M - M0) / np.linalg.norm(M0)
        
        def loss(pi, pi_sample, reg_para):
            ans = -np.sum(pi_sample * np.log(pi)) \
                + reg_para * (ot.rot(C1, pi.sum(axis=1), pi_sample.sum(axis=1))[-1] + ot.rot(C2, pi.sum(axis=0), pi_sample.sum(axis=0))[-1])
            return ans

        
        losses = []
        KLs = []
        constraints = []
        best_loss = np.inf
        best_configuration = None
        

        for i in range(max_outer_iteration):
            Z = np.exp(- lam * C)
            M = delta * (np.outer(v, np.ones(self.n)) + np.outer(np.ones(self.m), w)) * Z 
      
            for j in range(max_inner_iteration):
                def p(theta):
                    xi1 = (r_sample / (M - theta * Z).dot(eta))
                    return xi1.dot(Z).dot(eta) - 1
                
                def q(theta):
                    return xi.dot(Z).dot(c_sample / (M - theta * Z).T.dot(xi)) - 1
                
                theta0 = np.min(M.dot(eta) / Z.dot(eta))
                theta1 = scipy.optimize.root(p, theta0-10).x[0]
                xi = r_sample / (M - theta1 * Z).dot(eta)
        
                theta0 = np.min(M.dot(eta) / Z.dot(eta))
                theta2 = scipy.optimize.root(q, theta0-10).x[0]
                eta = c_sample / (M - theta2 * Z).T.dot(xi)
                
            pi = np.dot(np.diag(xi), np.exp(-lam * C) * eta)

            grad_C = lam * (self.pi_sample + (theta1 - delta*(np.outer(v, np.ones(self.n)) + np.outer(np.ones(self.m), w))) * pi)

            factor = grad_C * degree * gamma * np.power(gamma * self.U0.T.dot(A).dot(self.V0) + const , degree - 1)
            grad_A = self.U0.dot(factor).dot(self.V0.T)
            A -= learning_rate * grad_A
            C = np.power(gamma * self.U0.T.dot(A).dot(self.V0) + const, degree)
        
            v = np.log(ot.rot(C1, pi.sum(axis=1), r_sample, lambda_mu)[2]) / lambda_mu
            w = np.log(ot.rot(C2, pi.sum(axis=0), c_sample, lambda_nu)[2]) / lambda_nu
            v_dual = (np.log(pi.sum(axis=1)) - np.log(np.sum(np.exp(lambda_mu * (np.outer(np.ones(self.m), v) - C1)), axis=0))) / lambda_mu
            w_dual = (np.log(pi.sum(axis=0)) - np.log(np.sum(np.exp(lambda_nu * (np.outer(np.ones(self.n), w) - C2)), axis=0))) / lambda_nu
            
            losses.append(loss(pi, self.pi_sample, delta))
            KLs.append(KL(pi, self.pi_sample))
            
            if KLs[i] < best_loss:
                best_configuration = [C, A, pi]
                best_loss = KLs[i]
        
        self.A = A
        self.C = C
        return best_configuration
        
