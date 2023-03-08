import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from algorithm_made import Made
import numpy as np



def generate_model_inverse_maf(setting):


    hidden_shape = setting["z_hidden_shape"]
    n_layers = setting["z_n_layers"]
    input_shape = setting["z_dimension"] 

    permutation = tf.cast(np.concatenate((np.arange(input_shape/2,input_shape),np.arange(0,input_shape/2))), tf.int32)[:input_shape]

    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(input_shape), scale_diag=tf.ones(input_shape))  # specify base distribution

    bijectors = []

    for _ in range(0, n_layers):
        bijectors.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = \
            Made(params=2, hidden_units=hidden_shape, activation="relu"))))
        bijectors.append(tfb.Permute(permutation=permutation))  # data permutation after layers of MAF
      
        #bijectors.append(tfb.Permute(permutation=[1, 2, 0]))  # data permutation after layers of MAF
    
    bijector = tfb.Chain(bijectors=list(reversed(bijectors)), name='chain_of_model')

    model = tfd.TransformedDistribution(
      distribution= base_dist,
      bijector=bijector
    )
    return model


