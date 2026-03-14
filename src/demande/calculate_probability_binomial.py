import scipy 
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions


def calculate_probability_binomial(X_plot):
    mvn = tfd.MultivariateNormalDiag(
        loc=[1., -1],
        scale_diag=[1, 2.])
    return mvn.prob(X_plot)
