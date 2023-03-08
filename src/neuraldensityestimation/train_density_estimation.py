import tensorflow as tf

def train_density_estimation(distribution, optimizer, batch):
    """
    Train function for density estimation normalizing flows.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
    :param batch: Batch of the train data.
    :return: loss.
    """
    #tf.print("hola")
    #tf.print(batch.shape)
    with tf.GradientTape() as tape:
        tape.watch(distribution.trainable_variables)
        #tf.print(distribution.log_prob(batch))
        loss = -tf.reduce_mean(distribution.log_prob(batch))  # negative log likelihood
    #tf.print("gradient")
    gradients = tape.gradient(loss, distribution.trainable_variables)
    optimizer.apply_gradients(zip(gradients, distribution.trainable_variables))

    return loss


