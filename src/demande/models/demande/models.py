'''
Quantum Measurement Classfiication Models
'''

import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from demande.models.demande import layers



class QMDensity(tf.keras.Model):
    """
    A Quantum Measurement Density Estimation model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        dim_x: dimension of the input quantum feature map
    """
    def __init__(self, fm_x, dim_x):
        super(QMDensity, self).__init__()
        self.fm_x = fm_x
        self.dim_x = dim_x
        self.qmd = layers.QMeasureDensity(dim_x)
        self.cp = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=0.,
            trainable=False     
            )

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = self.qmd(psi_x)
        return probs

    @tf.function
    def call_train(self, x):
        if not self.qmd.built:
            self.call(x)
        psi = self.fm_x(x)
        rho = self.cp([psi, tf.math.conj(psi)])
        num_samples = tf.cast(tf.shape(x)[0], rho.dtype)
        rho = tf.reduce_sum(rho, axis=0)
        self.num_samples.assign_add(num_samples)
        return rho

    def train_step(self, data):
        data =  data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        if x.shape[1] is not None:
            rho = self.call_train(x)
            self.qmd.weights[0].assign_add(rho)
        return {}

    def fit(self, *args, **kwargs):
        result = super(QMDensity, self).fit(*args, **kwargs)
        self.qmd.weights[0].assign(self.qmd.weights[0] / self.num_samples)
        return result

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class QMDensitySGD(tf.keras.Model):
    """
    A Quantum Measurement Density Estimation modeltrainable using
    gradient descent.
    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, num_eig=0, gamma=1, random_state=None):
        super(QMDensitySGD, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qmd = layers.QMeasureDensityEig(dim_x=dim_x, num_eig=num_eig)
        self.num_eig = num_eig
        self.dim_x = dim_x
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = self.qmd(psi_x)
        self.add_loss(-tf.reduce_sum(tf.math.log(probs)))
        return probs

    def set_rho(self, rho):
        return self.qmd.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}