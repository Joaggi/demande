import tensorflow as tf
import tensorflow_probability as tfp
import logging
import numpy as np


def load_potential_2(train_size, test_size, dimension=2):

    logging.debug(f"loading potential_2 trian_size: {train_size} test_size: {test_size} dimension: {dimension}")

    X = np.load("data/NF2/nf2.npy").astype(np.float32)
    X_densities = np.loadtxt("data/NF2/NF2_densities.csv").astype(np.float32)

    X_train = X[:train_size, :]
    X_train_densities = X_densities[:train_size]
    X_test = X[train_size: train_size + test_size, :]
    X_test_densities = X_densities[train_size: train_size + test_size]

    return X_train, X_train_densities, X_test, X_test_densities


def generate_potential_2_dataset(train_size, test_size, dimension=2):

    logging.debug(f"generating potential_2 trian_size: {train_size} test_size: {test_size} dimension: {dimension}")

    tfd = tfp.distributions

    """
    Potential functions U(x) from Rezende et al. 2015
    p(z) is then proportional to exp(-U(x)).
    Since we log this value later in the optimized bound,
    no need to actually exp().
    """

    def w_1(z):
        return tf.sin((2 * np.pi * z[:, 0]) / 4)

    def pot_2(z):
        u = .5 * ((z[:, 1] - w_1(z)) / .4) ** 2
        return tf.exp(-u)/8.   

    sample_size = train_size + test_size

    @tf.function
    def unnormalized_log_prob(x):  
      x = tf.reshape(x, (1,-1))
      res = pot_2(x)
      return res

    # Initialize the HMC transition kernel.
    num_results = int(sample_size)
    num_burnin_steps = int(300)
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_prob,
            num_leapfrog_steps=200,
            #unrolled_leapfrog_steps=2,
            state_gradients_are_stopped=False,
            step_size=np.array([0.01, 0.01])),
        num_adaptation_steps=int(num_burnin_steps * 0.9),
          target_accept_prob = 0.75)

    # Run the chain (with burn-in).
    @tf.function
    def run_chain():
      samples, is_accepted = tfp.mcmc.sample_chain(
          num_results=num_results,
          num_burnin_steps=num_burnin_steps,
          current_state=tf.constant([[-2.0,0.0]]),
          kernel=adaptive_hmc,
          trace_fn=lambda _, pkr: [pkr.inner_results.is_accepted, 
                                   #pkr.inner_results.accepted_results.step_size,
                                 pkr.inner_results.log_accept_ratio])
      return samples, is_accepted

    samples, is_accepted = run_chain()

    X_densities = np.exp(pot_2(X).numpy())/8.


    np.savetxt(fname="data/NF2/NF2.csv", delimiter=" ", X = samples)
    np.savetxt(fname="data/NF2/NF2_densities.csv", delimiter=" ", X = X_densities)



