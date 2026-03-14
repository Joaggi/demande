import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

## Build the Normalizing Flow

# The next step is to build the respective normalizing flow. In this example the Masked Autogregressive 
# Flow (MAF) implementation from Tensorflow is used. The documentation can be found here 1. 
# Other flows can be imported from "normalizingflows/flow_catalog.py". In Tensorflow several 
# layers of normalizing flows are chained with the "tfb.Chain" function 2. For flows that use 
# coupling transformations, e.g. Real NVP, Neural Spline Flow, and MAF, it is recommended to use
# permutation between the layers of the flow. Batch normalization, which is also implemented 
# in the "flow_catalog.py" file, improves the results on high dimensional datasets such as image data. 
# The normalzing flow is created by inputting the chained bijector with all layers a base distribution, 
# which is usually a Gaussian distribution, into the "tfd.TransformedDistribution" 3.


#%%

tfk = tf.keras
class Made(tfk.layers.Layer):
    """
    Implementation of a Masked Autoencoder for Distribution Estimation (MADE) [Germain et al. (2015)].
    The existing TensorFlow bijector "AutoregressiveNetwork" is used. The output is reshaped to output one shift vector
    and one log_scale vector.
    :param params: Python integer specifying the number of parameters to output per input.
    :param event_shape: Python list-like of positive integers (or a single int), specifying the shape of the input to this layer, which is also the event_shape of the distribution parameterized by this layer. Currently only rank-1 shapes are supported. That is, event_shape must be a single integer. If not specified, the event shape is inferred when this layer is first called or built.
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    :param activation: An activation function. See tf.keras.layers.Dense. Default: None.
    :param use_bias: Whether or not the dense layers constructed in this layer should have a bias term. See tf.keras.layers.Dense. Default: True.
    :param kernel_regularizer: Regularizer function applied to the Dense kernel weight matrices. Default: None.
    :param bias_regularizer: Regularizer function applied to the Dense bias weight vectors. Default: None.
    """

    def __init__(self, params, event_shape=None, hidden_units=None, activation=None, use_bias=True,
                 kernel_regularizer=None, bias_regularizer=None, name="made"):

        super(Made, self).__init__(name=name)

        self.params = params
        self.event_shape = event_shape
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.network = tfb.AutoregressiveNetwork(params=params, event_shape=event_shape, hidden_units=hidden_units,
                                                 activation=activation, use_bias=use_bias, kernel_regularizer=kernel_regularizer, 
                                                 bias_regularizer=bias_regularizer)

    def call(self, x):
        #print(self.network(x))
        shift, log_scale = tf.unstack(self.network(x), num=2, axis=-1)

        return shift, tf.math.tanh(log_scale)



