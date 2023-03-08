import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np

import qmc.tf.layers as layers
import qmc.tf.models as models

def generate_model_dmkde_sgd(setting):

    sigma = setting["z_sigma"]
    gamma= 1/ (2*sigma**2)
    
    polinomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(setting["z_base_lr"], \
            setting["z_decay_steps"], setting["z_end_lr"], power=setting["z_power"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=polinomial_decay)  # optimizer

    num_eig = int(setting["z_dim_eig"] * setting["z_dim_rff"])

    dmkde_trainable_sgd = models.QMDensitySGD(setting["z_dimension"],  setting["z_dim_rff"], num_eig=num_eig,\
                    gamma=gamma, random_state=setting["z_random_state"])
    dmkde_trainable_sgd.layers[0].trainable = setting["z_trainable_layers_0"]
    dmkde_trainable_sgd.layers[1].trainable = setting["z_trainable_layers_1"]
    dmkde_trainable_sgd.compile(optimizer)

    return dmkde_trainable_sgd

