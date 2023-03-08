import scipy 
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf


class Swiss_roll_sampler(object):
    def __init__(self, N, theta=2*np.pi, scale=2, sigma=0.4):
        np.random.seed(1024)
        self.total_size = N
        self.theta = theta
        self.scale = scale
        self.sigma = sigma
        params = np.linspace(0,self.theta,self.total_size)
        self.X_center = np.vstack((params*np.sin(scale*params),params*np.cos(scale*params)))
        self.X = self.X_center.T + np.random.normal(0,sigma,size=(self.total_size,2))
        np.random.shuffle(self.X)
        self.X_train, self.X_val,self.X_test = self.split(self.X)
        self.Y = None
        self.mean = 0
        self.sd = 0

    def split(self,data):
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
        
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_density(self,x):
        assert len(x)==2
        a = 1./(np.sqrt(2*np.pi)*self.sigma)
        de = 2*self.sigma**2
        nu = -np.sum((np.tile(x,[self.total_size,1])-self.X_center.T)**2  ,axis=1)
        return np.mean(a*np.exp(nu/de))

    def load_all(self):
        return self.X, self.Y


def calculate_probability_swiss_roll(X_plot):

    swiss_roll_sampler = Swiss_roll_sampler(X_plot.shape[0])
    return np.array([swiss_roll_sampler.get_density(x) for x in X_plot])
        
