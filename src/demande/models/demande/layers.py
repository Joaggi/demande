"""
Layers implementing quantum feature maps, measurements and
utilities.
"""

import numpy as np
import tensorflow as tf
from sklearn.kernel_approximation import RBFSampler
from demande.models.demande import _RBFSamplerORF



##### Quantum Feature Map Layers


class QFeatureMapRFF(tf.keras.layers.Layer):
    """Quantum feature map using random Fourier Features.
    Uses `RBFSampler` from sklearn to approximate an RBF kernel using
    random Fourier features.

    Input shape:
        (batch_size, dim_in)
    Output shape:
        (batch_size, dim)
    Arguments:
        input_dim: dimension of the input
        dim: int. Number of dimensions to represent a sample.
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """

    def __init__(
            self,
            input_dim: int,
            dim: int = 100,
            gamma: float = 1,
            random_state=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state


    def build(self, input_shape):
        rbf_sampler = RBFSampler(
            gamma=self.gamma,
            n_components=self.dim,
            random_state=self.random_state)
        x = np.zeros(shape=(1, self.input_dim))
        rbf_sampler.fit(x)
        self.rff_weights = tf.Variable(
            initial_value=rbf_sampler.random_weights_,
            dtype=tf.float32,
            trainable=True,
            name="rff_weights")
        self.offset = tf.Variable(
            initial_value=rbf_sampler.random_offset_,
            dtype=tf.float32,
            trainable=True,
            name="offset")
        self.built = True

    def call(self, inputs):
        vals = tf.matmul(inputs, self.rff_weights) + self.offset
        vals = tf.cos(vals)
        vals = vals * tf.sqrt(2. / self.dim)
        norms = tf.linalg.norm(vals, axis=-1)
        psi = vals / tf.expand_dims(norms, axis=-1)
        return psi

    def get_config(self):
        config = {
            "input_dim": self.input_dim,
            "dim": self.dim,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)



class QMeasureDensity(tf.keras.layers.Layer):
    """Quantum measurement layer for density estimation.

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, 1)
    Arguments:
        dim_x: int. the dimension of the input  state
    """

    def __init__(
            self,
            dim_x: int,
            **kwargs
    ):
        self.dim_x = dim_x
        super().__init__(**kwargs)

    def build(self, input_shape):
        if (not input_shape[1] is None) and input_shape[1] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x})')
        self.rho = self.add_weight(
            "rho",
            shape=(self.dim_x, self.dim_x),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        oper = tf.einsum(
            '...i,...j->...ij',
            inputs, tf.math.conj(inputs),
            optimize='optimal') # shape (b, nx, nx)
        rho_res = tf.einsum(
            '...ik, km, ...mi -> ...',
            oper, self.rho, oper,
            optimize='optimal')  # shape (b, nx, ny, nx, ny)
        return rho_res

    def compute_output_shape(self, input_shape):
        return (1,)

class QMeasureDensityEig(tf.keras.layers.Layer):
    """Quantum measurement layer for density estimation.
    Represents the density matrix using a factorization:
    
    `dm = tf.matmul(V, tf.transpose(V, conjugate=True))`

    This rerpesentation is ameanable to gradient-based learning

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, 1)
    Arguments:
        dim_x: int. the dimension of the input state
        num_eig: Number of eigenvectors used to represent the density matrix
    """

    def __init__(
            self,
            dim_x: int,
            num_eig: int =0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        if num_eig < 1:
            num_eig = dim_x
        self.num_eig = num_eig

    def build(self, input_shape):
        if (not input_shape[1] is None) and input_shape[1] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x})')
        self.eig_vec = self.add_weight(
            "eig_vec",
            shape=(self.dim_x, self.num_eig),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True)
        self.eig_val = self.add_weight(
            "eig_val",
            shape=(self.num_eig,),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        norms = tf.expand_dims(tf.linalg.norm(self.eig_vec, axis=0), axis=0)
        eig_vec = self.eig_vec / norms
        eig_val = tf.keras.activations.relu(self.eig_val)
        eig_val = eig_val / tf.reduce_sum(eig_val)
        rho_h = tf.matmul(eig_vec,
                          tf.linalg.diag(tf.sqrt(eig_val)))
        rho_h = tf.matmul(tf.math.conj(inputs), rho_h)
        rho_res = tf.einsum(
            '...i, ...i -> ...',
            rho_h, tf.math.conj(rho_h), 
            optimize='optimal') # shape (b,)
        return rho_res

    def set_rho(self, rho):
        """
        Sets the value of self.rho_h using an eigendecomposition.

        Arguments:
            rho: a tensor of shape (dim_x, dim_x)
        Returns:
            e: list of eigenvalues in non-decreasing order
        """
        if (len(rho.shape.as_list()) != 2 or
                rho.shape[0] != self.dim_x or
                rho.shape[1] != self.dim_x):
            raise ValueError(
                f'rho shape must be ({self.dim_x}, {self.dim_x})')
        if not self.built:
            self.build((None, self.dim_x))        
        e, v = tf.linalg.eigh(rho)
        self.eig_vec.assign(v[:, -self.num_eig:])
        self.eig_val.assign(e[-self.num_eig:])
        return e

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_eig ": self.num_eig
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (1,)

def complex_initializer(base_initializer):
    """
    Complex Initializer to use in ComplexQMeasureDensityEig
    taken from https://github.com/tensorflow/tensorflow/issues/17097
    """
    f = base_initializer()

    def initializer(*args, dtype=tf.complex64, **kwargs):
        real = f(*args, **kwargs)
        imag = f(*args, **kwargs)
        return tf.complex(real, imag)

    return initializer

class ComplexQMeasureDensity(tf.keras.layers.Layer):
    """Quantum measurement layer for density estimation with complex values.
    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, 1)
    Arguments:
        dim_x: int. the dimension of the input  state
    """

    def __init__(
            self,
            dim_x: int,
            **kwargs
    ):
        self.dim_x = dim_x
        super().__init__(**kwargs)

    def build(self, input_shape):
        if (not input_shape[1] is None) and input_shape[1] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x})')
        with tf.device('cpu:0'):
            self.rho = self.add_weight(
                "rho",
                shape=(self.dim_x, self.dim_x),
                dtype=tf.complex64,
                initializer=complex_initializer(tf.keras.initializers.Zeros),
                trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        rho_res = tf.einsum(
            '...k, km, ...m -> ...',
            tf.math.conj(inputs), self.rho, inputs,
            optimize='optimal')  # shape (b,)
        return rho_res

    def compute_output_shape(self, input_shape):
        return (1,)

class ComplexQMeasureDensityEig(tf.keras.layers.Layer):
    """Quantum measurement layer for density estimation with complex terms.
    Represents the density matrix using a factorization:

    `dm = tf.matmul(V, tf.transpose(V, conjugate=True))`

    This rerpesentation is ameanable to gradient-based learning

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, 1)
    Arguments:
        dim_x: int. the dimension of the input state
        num_eig: Number of eigenvectors used to represent the density matrix
    """

    def __init__(
            self,
            dim_x: int,
            num_eig: int =0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        if num_eig < 1:
            num_eig = dim_x
        self.num_eig = num_eig

    def build(self, input_shape):
        if (not input_shape[1] is None) and input_shape[1] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x})')
        with tf.device('cpu:0'):
            self.eig_vec = self.add_weight(
                "eig_vec",
                shape=(self.dim_x, self.num_eig),
                dtype=tf.complex64,
                initializer=complex_initializer(tf.random_normal_initializer),
                trainable=True)
        self.eig_val = self.add_weight(
            "eig_val",
            shape=(self.num_eig,),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.complex64)
        norms = tf.expand_dims(tf.linalg.norm(self.eig_vec, axis=0), axis=0)
        eig_vec = self.eig_vec / norms
        eig_val = tf.keras.activations.relu(self.eig_val)
        eig_val = eig_val / tf.reduce_sum(eig_val)
        rho_h = tf.matmul(eig_vec,
                          tf.cast(tf.linalg.diag(tf.sqrt(eig_val)), 
                                  tf.complex64))
        rho_h = tf.matmul(tf.math.conj(inputs), rho_h)
        rho_res = tf.einsum(
            '...i, ...i -> ...',
            rho_h, tf.math.conj(rho_h), 
            optimize='optimal') # shape (b,)
        rho_res = tf.cast(rho_res, tf.float32)
        return rho_res

    def set_rho(self, rho):
        """
        Sets the value of self.rho_h using an eigendecomposition.

        Arguments:
            rho: a tensor of shape (dim_x, dim_x)
        Returns:
            e: list of eigenvalues in non-decreasing order
        """
        if (len(rho.shape.as_list()) != 2 or
                rho.shape[0] != self.dim_x or
                rho.shape[1] != self.dim_x):
            raise ValueError(
                f'rho shape must be ({self.dim_x}, {self.dim_x})')
        if not self.built:
            self.build((None, self.dim_x))        
        e, v = tf.linalg.eigh(rho)
        self.eig_vec.assign(v[:, -self.num_eig:])
        self.eig_val.assign(e[-self.num_eig:])
        return e

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_eig ": self.num_eig
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (1,)



class CrossProduct(tf.keras.layers.Layer):
    """Calculates the cross product of 2 inputs.

    Input shape:
        A list of 2 tensors [t1, t2] with shapes
        (batch_size, n) and (batch_size, m)
    Output shape:
        (batch_size, n, m)
    Arguments:
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)


    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('A `CrossProduct` layer should be called '
                             'on a list of 2 inputs.')
        if len(input_shape[0]) > 11 or len(input_shape[1]) > 11:
            raise ValueError('Input tensors cannot have more than '
                             '11 dimensions.')
        idx1 = 'abcdefghij'
        idx2 = 'klmnopqrst'
        self.eins_eq = ('...' + idx1[:len(input_shape[0]) - 1] + ',' +
                        '...' + idx2[:len(input_shape[1]) - 1] + '->' +
                        '...' + idx1[:len(input_shape[0]) - 1] +
                        idx2[:len(input_shape[1]) - 1])
        self.built = True

    def call(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `CrossProduct` layer should be called '
                             'on exactly 2 inputs')
        cp = tf.einsum(self.eins_eq,
                       inputs[0], inputs[1], optimize='optimal')
        return cp

    def compute_output_shape(self, input_shape):
        return (input_shape[0][1], input_shape[1][1])
