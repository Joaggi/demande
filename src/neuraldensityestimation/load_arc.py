import tensorflow_probability as tfp
import logging
import numpy as np
import tensorflow as tf

def load_arc(train_size, test_size, dimension=2):
    logging.debug(f"loading arc trian_size: {train_size} test_size: {test_size} dimension: {dimension}")

    X_train = np.load("data/arc/arc_train.npy").astype(np.float32)[:train_size, :dimension]
    X_train_densities = np.load("data/arc/arc_train_density.npy").astype(np.float32)[:train_size] 
    X_test = np.load("data/arc/arc_test.npy").astype(np.float32)[:test_size, :dimension]
    X_test_densities = np.load("data/arc/arc_test_density.npy").astype(np.float32)[:test_size]

    return X_train, X_train_densities, X_test, X_test_densities


def generate_arc_dataset(train_size, test_size, dimension=2):

    logging.debug(f"generating arc trian_size: {train_size} test_size: {test_size} dimension: {dimension}")

    dataset_size = train_size + test_size

    tfd = tfp.distributions

    x2_dist = tfd.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample(dataset_size)
    x1 = tfd.Normal(loc=.25 * tf.square(x2_samples),
                scale=tf.ones(dataset_size, dtype=tf.float32))
    x1_samples = x1.sample()
    x_samples = tf.stack([x1_samples, x2_samples], axis=1)    

    X_densities = x2_dist.prob(x_samples[:,1]) * x1.prob(x_samples[:,0])  

    X_train = x_samples[:train_size,:]
    X_train_density = X_densities.numpy()[:train_size]
    X_test = x_samples[train_size:,:]
    X_test_density = X_densities.numpy()[train_size:]

    np.save("data/arc/arc_train.npy", X_train)
    np.save("data/arc/arc_train_density.npy", X_train_density)
    np.save("data/arc/arc_test.npy", X_test)
    np.save("data/arc/arc_test_density.npy", X_test_density)




if __name__ == "__main__":

    generate_arc_dataset(train_size=1000000, test_size=10000, dimension=2)
 
