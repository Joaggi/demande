import tensorflow_probability as tfp
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os 
import math

def load_star_eight(train_size, test_size, dimension=2):
    logging.debug(f"loading star_eight trian_size: {train_size} test_size: {test_size} dimension: {dimension}")

    X_train = np.load("data/star_eight/star_eight_train.npy").astype(np.float32)[:train_size, :dimension] 
    X_train_densities = np.load("data/star_eight/star_eight_train_density.npy").astype(np.float32) [:train_size] 
    X_test = np.load("data/star_eight/star_eight_test.npy").astype(np.float32)[:test_size, :dimension]
    X_test_densities = np.load("data/star_eight/star_eight_test_density.npy").astype(np.float32)[:test_size] 
    return X_train, X_train_densities, X_test, X_test_densities


def generate_star_eight_dataset(train_size, test_size, dimension=2):

    logging.debug(f"generating star_eight trian_size: {train_size} test_size: {test_size} dimension: {dimension}")

    tfd = tfp.distributions
          
    repeat_generation_for_small_sample(train_size, "data/star_eight/star_eight_train")
    repeat_generation_for_small_sample(test_size, "data/star_eight/star_eight_test")


def repeat_generation_for_small_sample(sample_size, path):
    for _ in range(int(sample_size/10000)+1):
      total_data, total_density = generate_samples(10000)
      print(total_data.shape)
      print(total_density.shape)

      if(os.path.isfile(f"{path}.npy")):
        total_data_saved = np.load(f"{path}.npy")
        total_density_saved = np.load(f"{path}_density.npy")
        total_density = np.hstack([total_density, total_density_saved])
        total_data = np.vstack([total_data, total_data_saved])

      else:
        total_data_saved = total_data
        total_density_saved = total_density

      print(total_density.shape)
      print(total_data.shape)
      np.save(f"{path}.npy", total_data)
      np.save(f"{path}_density.npy", total_density)  


def generate_samples(sample_size):

    n_components = 8
    def cal_cov(theta,sx=1,sy=0.4**2):
        Scale = np.array([[sx, 0], [0, sy]])
        c, s = np.cos(theta), np.sin(theta)
        Rot = np.array([[c, -s], [s, c]])
        T = Rot.dot(Scale)
        Cov = T.dot(T.T)
        return Cov
    radius = 3
    mean = np.array([[radius*math.cos(2*np.pi*idx/float(n_components)),radius*math.sin(2*np.pi*idx/float(n_components))] for idx in range(n_components)])
    cov = np.array([cal_cov(2*np.pi*idx/float(n_components)) for idx in range(n_components)])

    multivariate = [tfp.distributions.MultivariateNormalFullCovariance(
        loc=star_mean.astype("float32"),
         covariance_matrix=star_cov.astype("float32")) for star_mean, star_cov in zip(mean, cov)]

    star_multivariate = tfp.distributions.Mixture(
       cat = tfp.distributions.Categorical(probs=[1/8 for _ in range(8)]),
       components=multivariate)


    datos = tf.cast(star_multivariate.sample(sample_size), tf.float64).numpy()

    densities = star_multivariate.prob(datos.astype("float32")) 

    return datos, densities

if __name__ == "__main__":
    
    train_size = 1000000
    test_size = 10000

    generate_star_eight_dataset(train_size=train_size, test_size=test_size, dimension=2)

