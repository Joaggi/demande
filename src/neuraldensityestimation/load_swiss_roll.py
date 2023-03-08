import tensorflow_probability as tfp
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_swiss_roll(train_size, test_size, dimension=2):
    logging.debug(f"loading swiss_roll trian_size: {train_size} test_size: {test_size} dimension: {dimension}")

    X_train = np.load("data/swiss_roll/swiss_roll_train.npy").astype(np.float32)[:train_size, :dimension]
    X_train_densities = np.load("data/swiss_roll/swiss_roll_train_density.npy").astype(np.float32)[:train_size]
    X_test = np.load("data/swiss_roll/swiss_roll_test.npy").astype(np.float32)[:test_size, :dimension]
    X_test_densities = np.load("data/swiss_roll/swiss_roll_test_density.npy").astype(np.float32)[:test_size]
    return X_train, X_train_densities, X_test, X_test_densities


def generate_swiss_roll_dataset(train_size, test_size, dimension=2):

    logging.debug(f"generating swiss_roll trian_size: {train_size} test_size: {test_size} dimension: {dimension}")

    tfd = tfp.distributions
    
    X_train, X_train_density = generate_samples(train_size)
    X_test, X_test_density = generate_samples(test_size)

    np.save("data/swiss_roll/swiss_roll_train.npy", X_train)
    np.save("data/swiss_roll/swiss_roll_train_density.npy", X_train_density)
    np.save("data/swiss_roll/swiss_roll_test.npy", X_test)
    np.save("data/swiss_roll/swiss_roll_test_density.npy", X_test_density)


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


def generate_samples(number_of_points):
    logging.debug("Generating new samples")
    swiss_roll_sampler = Swiss_roll_sampler(number_of_points)
    train_set = swiss_roll_sampler.load_all()[0]
    logging.debug("Generating scatter plot")
    plt.scatter(train_set[:,0], train_set[:,1])
    logging.debug("Starting calculation of density")
    true_density = np.array([swiss_roll_sampler.get_density(x) for x in train_set])
    logging.debug(f"Calculation of density finished shape {true_density.shape}")

    return train_set, true_density

if __name__ == "__main__":

    generate_swiss_roll_dataset(train_size=100000, test_size=10000, dimension=2)
 
