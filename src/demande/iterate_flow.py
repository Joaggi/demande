import tensorflow as tf
import time
from utils.train_utils import train_density_estimation, nll
import numpy as np
import mlflow_wrapper
   

def iterate_flow(model, max_epochs, batched_train_data, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow):

  min_val_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)  # high value to ensure that first loss < min_loss
  global_step = []
  train_losses = [] 
  val_losses = []
  min_val_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)  # high value to ensure that first loss < min_loss
  min_train_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)
  min_val_epoch = 0
  min_train_epoch = 0
  delta_stop = 100  # threshold for early stopping

  @tf.function
  def train_density_estimation_tf_function(distribution, optimizer, batch):
    with tf.GradientTape() as tape:
        tape.watch(distribution.trainable_variables)
        loss = -tf.reduce_mean(distribution.log_prob(batch))  # negative log likelihood
    gradients = tape.gradient(loss, distribution.trainable_variables)
    optimizer.apply_gradients(zip(gradients, distribution.trainable_variables))

    return loss

  def train_density_no_tf(distribution, optimizer, batch):
    """
    Train function for density estimation normalizing flows without tf.function decorator
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
    :param batch: Batch of the train data.
    :return: loss.
    """
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(distribution.log_prob(batch)) # negative log likelihood
        gradients = tape.gradient(loss, distribution.trainable_variables)
        optimizer.apply_gradients(zip(gradients, distribution.trainable_variables))
        return loss

  t_start = time.time()  # start time
  val_loss = np.inf

  # start training
  for i in range(max_epochs):
      for batch in batched_train_data:
          #train_loss = train_density_estimation_tf_function(model, opt, batch)
          train_loss = train_density_no_tf(model, opt, batch)
      
      if algorithm == "planar_flow":
          for bijector in model.bijector.bijectors:
              bijector._u()  

      if i % int(100) == 0:
          
          val_loss = nll(model, test_set)
          global_step.append(i)
          train_losses.append(train_loss)
          val_losses.append(val_loss)
          print(f"{i}, train_loss: {train_loss}, val_loss: {val_loss}")

          if train_loss < min_train_loss:
              min_train_loss = train_loss
              min_train_epoch = i
              mlflow_wrapper.log_metric(mlflow, "train_loss", train_loss.numpy())

          if val_loss < min_val_loss:
              min_val_loss = val_loss
              min_val_epoch = i
              checkpoint.write(file_prefix=checkpoint_path)  # overwrite best val model
              mlflow_wrapper.log_metric(mlflow, "val_loss", val_loss.numpy())

          elif i - min_val_epoch > delta_stop:  # no decrease in min_val_loss for "delta_stop epochs"
              break

      #if i % int(1000) == 0:
      #    # plot heatmap every 1000 epochs
      #    plot_heatmap_2d(model, -2.0, 50.0, -35.0, 15.0, mesh_count=200)
  if val_loss < min_val_loss:
      min_val_loss = val_loss
      checkpoint.write(file_prefix=checkpoint_path)  # overwrite best val model
      mlflow_wrapper.log_metric(mlflow, "val_loss", val_loss.numpy())


  train_time = time.time() - t_start


