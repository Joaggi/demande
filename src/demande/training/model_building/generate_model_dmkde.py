import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np


import demande.models.demande.layers as layers
import demande.models.demande.models as models

def generate_model_dmkde(sigma, input_dimension, dim_rff, random_state):

    gamma= 1/ (2*sigma**2)

    fm_x = layers.QFeatureMapRFF(input_dimension, dim_rff, gamma=gamma, random_state=random_state)
    dmkde = models.QMDensity(fm_x, dim_rff)
    dmkde.compile()

    return dmkde

