import scipy 
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf

"""
Potential functions U(x) from Rezende et al. 2015
p(z) is then proportional to exp(-U(x)).
Since we log this value later in the optimized bound,
no need to actually exp().
"""

def w_1(z):
    return tf.sin((2 * np.pi * z[:, 0]) / 4)


def w_2(z):
    return 3 * tf.exp(-.5 * ((z[:, 0] - 1) / .6) ** 2)


def sigma(x):
    return 1 / (1 + tf.exp(- x))


def w_3(z):
    return 3 * sigma((z[:, 0] - 1) / .3)



def pot_2(z):
    u = .5 * ((z[:, 1] - w_1(z)) / .4) ** 2
    return tf.exp(-u)/8.

def calculate_probability_potential_2(X_plot):

    return (tf.exp(pot_2(X_plot)))
        
