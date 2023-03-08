import tensorflow_probability as tfp
import logging
import numpy as np
import tensorflow as tf

def load_bimodal_l(train_size, test_size, dimension=2):
    logging.debug(f"loading bimodal_l trian_size: {train_size} test_size: {test_size} dimension: {dimension}")


    X_train = np.load("data/bimodal_l/bimodal_l_train.npy").astype(np.float32)[:train_size, :dimension]
    X_train_densities = np.load("data/bimodal_l/bimodal_l_train_density.npy").astype(np.float32) [:train_size]
    X_test = np.load("data/bimodal_l/bimodal_l_test.npy").astype(np.float32)[:test_size, :dimension]
    X_test_densities = np.load("data/bimodal_l/bimodal_l_test_density.npy").astype(np.float32)[:test_size]
    return X_train, X_train_densities, X_test, X_test_densities


def generate_bimodal_l_dataset(train_size, test_size, dimension=2):

    logging.debug(f"generating bimodal_l trian_size: {train_size} test_size: {test_size} dimension: {dimension}")
    
    dataset_size = train_size + test_size

    tfd = tfp.distributions

    first_normal = tfp.distributions.MultivariateNormalDiag(
    loc=[1., -1],
    scale_diag=[1, 2.])

    second_normal = tfp.distributions.MultivariateNormalDiag(
        loc=[-2., 2],
        scale_diag=[2, 1.])

    bimix_gauss = tfp.distributions.Mixture(
      cat = tfp.distributions.Categorical(probs=[0.5, 0.5]),
      components=[first_normal,second_normal]) 


    x_samples = bimix_gauss.sample(dataset_size)
    X_densities = bimix_gauss.prob(x_samples)

    X_train = x_samples[:train_size,:]
    X_train_density = X_densities.numpy()[:train_size]
    X_test = x_samples[train_size:,:]
    X_test_density = X_densities.numpy()[train_size:]

    np.save("data/bimodal_l/bimodal_l_train.npy", X_train)
    np.save("data/bimodal_l/bimodal_l_train_density.npy", X_train_density)
    np.save("data/bimodal_l/bimodal_l_test.npy", X_test)
    np.save("data/bimodal_l/bimodal_l_test_density.npy", X_test_density)




if __name__ == "__main__":
    generate_bimodal_l_dataset(train_size=40000, test_size=10000, dimension=2)
 
