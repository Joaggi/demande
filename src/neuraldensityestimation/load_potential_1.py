import tensorflow as tf
import tensorflow_probability as tfp
import logging
import numpy as np


def load_potential_1(train_size, test_size, dimension=2):

    logging.debug(f"loading potential_1 trian_size: {train_size} test_size: {test_size} dimension: {dimension}")

    X = np.loadtxt("data/NF1/NF1_1M.csv").astype(np.float32)
    X_densities = np.loadtxt("data/NF1/NF1_1M_densities.csv").astype(np.float32)

    X_train = X[:train_size, :]
    X_train_densities = X_densities[:train_size]
    X_test = X[train_size: train_size + test_size, :]
    X_test_densities = X_densities[train_size: train_size + test_size]

    return X_train, X_train_densities, X_test, X_test_densities


def generate_potential_1_dataset(train_size, test_size, dimension=2):

    logging.debug(f"generating potential_1 trian_size: {train_size} test_size: {test_size} dimension: {dimension}")

    tfd = tfp.distributions

    """
    Potential functions U(x) from Rezende et al. 2015
    p(z) is then proportional to exp(-U(x)).
    Since we log this value later in the optimized bound,
    no need to actually exp().
    """

 
    def pot_1(z):
        z_1, z_2 = z[:, 0], z[:, 1]
        norm = tf.sqrt(z_1 ** 2 + z_2 ** 2)
        outer_term_1 = .5 * ((norm - 2) / .4) ** 2
        inner_term_1 = tf.exp((-.5 * ((z_1 - 2) / .6) ** 2))
        inner_term_2 = tf.exp((-.5 * ((z_1 + 2) / .6) ** 2))
        outer_term_2 = tf.math.log(inner_term_1 + inner_term_2 + 1e-7)
        u = outer_term_1 - outer_term_2
        return - u

    sample_size = train_size + test_size

    @tf.function
    def unnormalized_log_prob(x):  
      x = tf.reshape(x, (1,-1))
      res = pot_1(x)
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

    X_densities = np.exp(pot_1(samples).numpy())/6.529

    np.savetxt(fname="data/NF1/NF1_1M.csv", delimiter=" ", X = samples)
    np.savetxt(fname="data/NF1/NF1_1M_densities.csv", delimiter=" ", X = X_densities)



