"""
Multitask learning
"""

import numpy as np
import pickle
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.extmath import safe_sparse_dot
from time import strftime, gmtime


class DAN2Regressor(object):

    def __init__(self, depth=10, bounds=(0, 500)):
        self.bounds = bounds
        self.depth = depth
        self.lin_predictor = LinearRegression(fit_intercept=True)
        self.coef_ = None
        self.multi_coef_ = [None]*3
        self.name = strftime('dan2model-' + str(depth) +
                             '-%Y-%b-%d-%H-%M-%S', gmtime())
        #self.lin_predictions = None

    """ Layer activation """

    def f(self, x):

        f = self.f_k
        A = self.A
        alpha = self.alpha
        a = self.a
        rows = f.shape[0]
        ''' check if intercept term should be placed first'''
        #Xn = np.hstack((a, A[0]*f, A[1]*np.cos(alpha*x), A[2]*np.sin(alpha*x)))
        Xn = a + A[0]*f + A[1]*np.cos(alpha*x) + A[2]*np.sin(alpha*x)
        return np.sum(Xn)

    """ Method to get alpha column for DAN2 """

    def compute_alpha(self, X):
        cols = X.shape[1]

        """ Create resultant vector of ones """
        # R = np.ones(cols)
        R = np.random.rand(cols)
        #print('R', R.shape)

        """ Compute dot product """
        X_dot_R = (np.dot(X, R))  # +1)
        #print('XdR', X_dot_R.shape)
        X_dot_R = X_dot_R.reshape((len(X),))
        #print('XdR', X_dot_R.shape)

        """ Compute X and R magnitudes """
        X_mag = np.sqrt(np.sum(np.square(X), axis=1))  # + 1*1 )
        R_mag = np.sqrt(np.sum(R**2))  # + 1*1)

        """ Compute arccosine """
        acos = np.arccos(X_dot_R / (X_mag * R_mag))
        #print('acos', acos.shape)

        return acos.reshape(len(acos), 1)

    """ Linear method """

    def linear_reg(self, X, y):
        self.model['lr'] = LinearRegression(fit_intercept=True).fit(X, y)
        return self.model['lr'].predict(X), self.model['lr'].coef_[0], self.model['lr'].intercept_

    ''' '''

    def build_X1(self, f, alpha):
        return np.hstack((f, np.cos(alpha), np.sin(alpha)))

    ''' '''

    def build_Xn(self, f, A, alpha, mu):
        rows = f.shape[0]
        if A is None and mu is None:
            X = np.hstack((f, np.cos(alpha), np.sin(alpha)))
            A = LinearRegression(fit_intercept=True).fit(X, y)

        return np.hstack((A[0]*f, A[1]*np.cos(alpha*mu), A[2]*np.sin(alpha*mu)))

    def logging(self, coef_):
        if self.coef_ is None:
            self.coef_ = coef_.reshape(1, 5)

        else:
            self.coef_ = np.vstack((self.coef_, coef_))

    def multi_logging(self, coef_, head_number):
        if self.multi_coef_[head_number] is None:
            self.multi_coef_[head_number] = coef_.reshape(1, 5)

        else:
            self.multi_coef_[head_number] = np.vstack(
                (self.multi_coef_[head_number], coef_))

    """ Fit method  """

    def fit(self, X, y):

        # Number of rows
        m = X.shape[0]

        # Get non-linear projection of input records
        alpha = self.compute_alpha(X)

        # Get linear model from n input cols
        self.lin_predictor.fit(X, y)
        f_k = self.lin_predictor.predict(X)
        self.lin_predictions = f_k
        """ Start fit algorithm """
        i = 1
        mu = np.random.random()  # 1
        while (i <= self.depth):
            if i == 1:
                Xn = self.build_X1(f_k, alpha)
                lr = LinearRegression(fit_intercept=True).fit(Xn, y)
                A = lr.coef_[0]
                a = lr.intercept_
                f_k = lr.predict(Xn)
            else:
                mu = self.minimize(f_k, A, a, alpha)
                # eventually override the build_X1 method
                Xn = self.build_Xn(f_k, A, alpha, mu)
                lr = LinearRegression(fit_intercept=True).fit(Xn, y)
                A = lr.coef_[0]
                a = lr.intercept_
                f_k = lr.predict(Xn)

            # Error metrics
            mse = self.mse(f_k, y, m)
            # pred = np.where(f_k >= 0.5, 1, 0)
            # acc=mean_squared_error(y,pred)
            # acc = accuracy_score(y, pred)

            # Save layer
            coef_ = A.reshape((1, 3))
            coef_ = np.insert(coef_, 0, a)
            coef_ = np.insert(coef_, 0, mu)
            print(i, coef_)
            self.logging(coef_)

            # add layers
            print('Iteration:', i, " Mu:", mu, "MSE:",
                  mse, 'fk:', f_k)  # , "Accuracy:", acc)

            i += 1
        return f_k, mse

    def minimize(self, f_k, A, a, alpha):
        self.f_k = f_k
        self.A = A
        self.alpha = alpha
        self.a = a
        res = minimize_scalar(self.f, bounds=self.bounds, method='bounded')
        return res.x

    def mse(self, f_k, y, m):
        return np.sum((f_k - y)**2) / m

    def _activation_function(self, X, coef_):
        intercept = coef_[0]
        A = coef_[1:]
        return safe_sparse_dot(X, A.T, dense_output=True) + intercept

    def predict(self, X_test, head_number):
        X = X_test
        m = X.shape[0]
        alpha = self.compute_alpha(X)
        f_k = self.lin_predictor.predict(X)
        i = 0

        for coef_ in self.coef_:
            mu = coef_[0]
            if i == 0:
                X = np.hstack((f_k, np.cos(alpha*mu), np.sin(alpha*mu)))
                f_k = self._activation_function(X, coef_[1:])
                f_k = f_k.reshape(m, 1)
            else:
                X = np.hstack((prev_coef_[2]*f_k, prev_coef_[3]*np.cos(alpha*mu), prev_coef_[4]*np.sin(alpha*mu)))
                f_k = self._activation_function(X, coef_[1:])
                f_k = f_k.reshape(m, 1)
            i += 1
            prev_coef_ = coef_
        i = 0
        for multi_coef_ in self.multi_coef_[head_number]:
            mu = multi_coef_[0]
            if i == 0:
                X = np.hstack((f_k, np.cos(alpha*mu), np.sin(alpha*mu)))
                f_k = self._activation_function(X, multi_coef_[1:])
                f_k = f_k.reshape(m, 1)
            else:
                X = np.hstack((prev_multi_coef_[2]*f_k, prev_multi_coef_[3]*np.cos(alpha*mu), prev_multi_coef_[4]*np.sin(alpha*mu)))
                f_k = self._activation_function(X, prev_multi_coef_[1:])
                f_k = f_k.reshape(m, 1)
            prev_multi_coef_ = multi_coef_
            i += 1

        return f_k

    def multihead(self, X, y, head_depth, head_number, prev_f_k):
        m = X.shape[0]
        alpha = self.compute_alpha(X)
        f_k, i = prev_f_k, 1
        print('-'*20, f_k)
        mu = np.random.random()
        while i <= head_depth:
            if i == 1:
                Xn = self.build_X1(f_k, alpha)
                lr = LinearRegression(fit_intercept=True).fit(Xn, y)
                A = lr.coef_[0]
                a = lr.intercept_
                f_k = lr.predict(Xn)
            else:
                mu = self.minimize(f_k, A, a, alpha)
                # eventually override the build_X1 method
                Xn = self.build_Xn(f_k, A, alpha, mu)
                lr = LinearRegression(fit_intercept=True).fit(Xn, y)
                A = lr.coef_[0]
                a = lr.intercept_
                f_k = lr.predict(Xn)
            mse = self.mse(f_k, y, m)
            multi_coef_ = A.reshape((1, 3))
            multi_coef_ = np.insert(multi_coef_, 0, a)
            multi_coef_ = np.insert(multi_coef_, 0, mu)
            self.multi_logging(multi_coef_, head_number)
            i += 1
        return mse, f_k
