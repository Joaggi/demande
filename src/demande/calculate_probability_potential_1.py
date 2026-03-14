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


def pot_1(z):
    z_1, z_2 = z[:, 0], z[:, 1]
    norm = tf.sqrt(z_1 ** 2 + z_2 ** 2)
    outer_term_1 = .5 * ((norm - 2) / .4) ** 2
    inner_term_1 = tf.exp((-.5 * ((z_1 - 2) / .6) ** 2))
    inner_term_2 = tf.exp((-.5 * ((z_1 + 2) / .6) ** 2))
    outer_term_2 = tf.math.log(inner_term_1 + inner_term_2 + 1e-7)
    u = outer_term_1 - outer_term_2
    return - u

def calculate_probability_potential_1(X_plot):

    return (tf.exp(pot_1(X_plot)))
        
