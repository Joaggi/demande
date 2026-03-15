import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np


import demande.models.demande.layers as layers
import demande.models.demande.models as models

from demande.configs.dmkde_sgd_config import DmkdeSgdOptimizerConfig, DmkdeSgdParameterConfig



def generate_model_dmkde_sgd(dmkde_sgd_parameters: DmkdeSgdParameterConfig, optimizer: DmkdeSgdOptimizerConfig):

#optimizer
#base_lr, decay_steps, end_lr, power, 

#models
#        sigma, eig_dim, rff_dim, input_dimension, random_state, layer_0_trainable, layer_1_trainable

    gamma= 1/ (2 * dmkde_sgd_parameters.sigma ** 2)
    
    polynomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(optimizer.base_lr, \
            optimizer.decay_steps, optimizer.end_lr, power=optimizer.power)
    optimizer = tf.keras.optimizers.Adam(learning_rate=polynomial_decay)  # optimizer

    num_eig = int(dmkde_sgd_parameters.eig_dim * dmkde_sgd_parameters.rff_dim)

    dmkde_trainable_sgd = models.QMDensitySGD(dmkde_sgd_parameters.input_dimension, dmkde_sgd_parameters.rff_dim, num_eig=num_eig, \
                                              gamma=gamma, random_state=dmkde_sgd_parameters.random_state)
    dmkde_trainable_sgd.layers[0].trainable = dmkde_sgd_parameters.layer_0_trainable
    dmkde_trainable_sgd.layers[1].trainable = dmkde_sgd_parameters.layer_1_trainable
    dmkde_trainable_sgd.compile(optimizer)

    return dmkde_trainable_sgd

