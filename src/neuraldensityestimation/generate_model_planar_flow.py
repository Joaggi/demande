import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np

from algorithm_planar_flow import PlanarFlow


def generate_model_planar_flow(setting):

    n_layers = setting["z_n_layers"]
    input_shape = setting["z_dimension"] 



    #permutation = tf.cast(np.concatenate((np.arange(input_shape/2,input_shape),np.arange(0,input_shape/2))), tf.int32)[:input_shape]
    # base distribution
    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(input_shape, tf.float32)) 
    # create a normalizing flow

    bijector_chain = []
    for _ in range(n_layers):
      bijector_chain.append(PlanarFlow(input_dimensions=input_shape, case="density_estimation") )
      #bijector_chain.append(tfp.bijectors.Permute(permutation))

    bijector = tfb.Chain(bijectors=list(reversed(bijector_chain)), name='chain_of_real_nvp')

    flow = tfd.TransformedDistribution(
      distribution=base_dist,
      bijector=bijector
    )

    # ensure invertibility
    for bijector in flow.bijector.bijectors:
      bijector._u()
    print(len(flow.trainable_variables))

    return flow

