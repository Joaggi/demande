import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np

import qmc.tf.layers as layers
import qmc.tf.models as models

def generate_model_dmkde(setting):

    sigma = setting["z_sigma"]
    gamma= 1/ (2*sigma**2)

    fm_x = layers.QFeatureMapRFF(setting["z_dimension"], dim=setting["z_dim_rff"], gamma=gamma, random_state=setting["z_random_state"])
    dmkde = models.QMDensity(fm_x, setting["z_dim_rff"])
    dmkde.compile()

    return dmkde

