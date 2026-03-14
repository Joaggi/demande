import scipy 
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions


def calculate_probability_arc(X):
    x2_dist = tfd.Normal(loc=0., scale=4.)
    return x2_dist.prob(X[:,1]) * scipy.stats.norm(0.25 * np.square(X[:,1]), 1).pdf(X[:,0])


