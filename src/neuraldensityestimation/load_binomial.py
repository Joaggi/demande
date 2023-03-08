import tensorflow_probability as tfp
import logging
import numpy as np

def load_binomial(train_size, test_size, dimension=2):

    logging.debug(f"loading binomial trian_size: {train_size} test_size: {test_size} dimension: {dimension}")

    X_train = np.load("data/binomial/binomial_train.npy")[:train_size, :dimension] 
    X_train_densities = np.load("data/binomial/binomial_train.npy")[:train_size]
    X_test = np.load("data/binomial/binomial_test.npy")[:test_size, :dimension]
    X_test_densities = np.load("data/binomial/binomial_test_density.npy")[:test_size]
    
    return X_train, X_train_densities, X_test, X_test_densities


def generate_binomial_dataset(train_size, test_size, dimension=2):

    logging.debug(f"generating binomial trian_size: {train_size} test_size: {test_size} dimension: {dimension}")

    tfd = tfp.distributions

    # Initialize a single 2-variate Gaussian.
    mvn = tfd.MultivariateNormalDiag(
        loc=[1., -1],
        scale_diag=[1, 2.])

    X_train = mvn.sample(sample_shape=train_size, seed = 1)
    X_test = mvn.sample(sample_shape=test_size, seed = 1)

    X_train_density = mvn.prob(X_train)
    X_test_density = mvn.prob(X_test)

    np.save("data/binomial/binomial_train.npy", X_train)
    np.save("data/binomial/binomial_train_density.npy", X_train_density)
    np.save("data/binomial/binomial_test.npy", X_test)
    np.save("data/binomial/binomial_test_density.npy", X_test_density)




if __name__ == "__main__":
    generate_binomial_dataset(train_size=40000, test_size=10000, dimension=2)
    
