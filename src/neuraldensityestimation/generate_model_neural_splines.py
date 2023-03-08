import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np

from algorithm_neural_spline_flow import NeuralSplineFlow
from generate_model_made import generate_model_made


def generate_model_neural_splines(setting):

    input_shape = setting["z_dimension"]
    n_layers = setting["z_n_layers"]
    b_interval = setting["z_b_interval"]
    hidden_shape = setting["z_hidden_shape"]
    number_of_bins = setting["z_number_of_bins"]



    permutation = tf.cast(np.concatenate((np.arange(input_shape/2,input_shape),np.arange(0,input_shape/2))), tf.int32)[:input_shape]
    #permutation = tf.cast(np.concatenate((np.arange(input_shape/2,input_shape),np.arange(0,input_shape/2))), tf.int32)
    

    #base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(input_shape), scale_diag=tf.ones(input_shape))  # specify base distribution
    base_dist = tfd.MultivariateNormalDiag(tf.zeros(input_shape, tf.float32))  # specify base distribution

    bijector_chain = []
    for _ in range(n_layers):
      bijector_chain.append(NeuralSplineFlow(input_dim=input_shape, d_dim=int(input_shape/2)+1, \
            number_of_bins=number_of_bins, nn_layers = hidden_shape, b_interval=b_interval))
      #bijector_chain.append(NeuralSplineFlow(input_dim=input_shape, d_dim=int(input_shape/2)+1, \
      #      number_of_bins=8,  b_interval=b_interval))


      bijector_chain.append(tfp.bijectors.Permute(permutation))

    bijector = tfb.Chain(bijectors=list(reversed(bijector_chain)), name='chain_of_real_nvp')

    flow = tfd.TransformedDistribution(
      distribution=base_dist,
      bijector=bijector
    )
    return flow



