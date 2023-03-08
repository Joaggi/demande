import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors




'''------------------------------------- Batch Normalization Bijector -----------------------------------------------'''


class BatchNorm(tfb.Bijector):
    """
    Implementation of a Batch Normalization layer for use in normalizing flows according to [Papamakarios et al. (2017)].
    The moving average of the layer statistics is adapted from [Dinh et al. (2016)].
    :param eps: Hyperparameter that ensures numerical stability, if any of the elements of v is near zero.
    :param decay: Weight for the update of the moving average, e.g. avg = (1-decay)*avg + decay*new_value.
    """

    def __init__(self, eps=1e-5, decay=0.95, validate_args=False, name="batch_norm"):
        super(BatchNorm, self).__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            validate_args=validate_args,
            name=name)

        self._vars_created = False
        self.eps = eps
        self.decay = decay

    def _create_vars(self, x):
        # account for 1xd and dx1 vectors
        if len(x.get_shape()) == 1:
            n = x.get_shape().as_list()[0]
        if len(x.get_shape()) == 2: 
            n = x.get_shape().as_list()[1]

        self.beta = tf.compat.v1.get_variable('beta', [1, n], dtype=tf.float32)
        self.gamma = tf.compat.v1.get_variable('gamma', [1, n], dtype=tf.float32)
        self.train_m = tf.compat.v1.get_variable(
            'mean', [1, n], dtype=tf.float32, trainable=False)
        self.train_v = tf.compat.v1.get_variable(
            'var', [1, n], dtype=tf.float32, trainable=False)

        self._vars_created = True

    def _forward(self, u):
        if not self._vars_created:
            self._create_vars(u)
        return (u - self.beta) * tf.exp(-self.gamma) * tf.sqrt(self.train_v + self.eps) + self.train_m

    def _inverse(self, x):
        # Eq. 22 of [Papamakarios et al. (2017)]. Called during training of a normalizing flow.
        if not self._vars_created:
            self._create_vars(x)

        # statistics of current minibatch
        m, v = tf.nn.moments(x, axes=[0], keepdims=True)
        
        # update train statistics via exponential moving average
        self.train_v.assign_sub(self.decay * (self.train_v - v))
        self.train_m.assign_sub(self.decay * (self.train_m - m))

        # normalize using current minibatch statistics, followed by BN scale and shift
        return (x - m) * 1. / tf.sqrt(v + self.eps) * tf.exp(self.gamma) + self.beta

    def _inverse_log_det_jacobian(self, x):
        # at training time, the log_det_jacobian is computed from statistics of the
        # current minibatch.
        if not self._vars_created:
            self._create_vars(x)
            
        _, v = tf.nn.moments(x, axes=[0], keepdims=True)
        abs_log_det_J_inv = tf.reduce_sum(
            self.gamma - .5 * tf.math.log(v + self.eps))
        return abs_log_det_J_inv


