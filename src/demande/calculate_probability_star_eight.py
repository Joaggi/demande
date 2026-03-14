import scipy 
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf
import math

def generate_samples(X_plot):

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



    densities = star_multivariate.prob(X_plot.astype("float32")) 

    return densities


def calculate_probability_star_eight(X_plot):

    return  generate_samples(X_plot)
        
