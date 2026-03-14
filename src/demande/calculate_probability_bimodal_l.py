import scipy 
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions


def calculate_probability_bimodal_l(X_plot):
    mix = 0.5

    first_normal = tfd.MultivariateNormalDiag(
    loc=[1., -1],
    scale_diag=[1, 2.])

    second_normal = tfd.MultivariateNormalDiag(
    loc=[-2., 2],
    scale_diag=[2, 1.])

    bimix_gauss = tfd.Mixture(
        cat=tfd.Categorical(probs=[mix, 1.-mix]),
        components=[
            first_normal,
            second_normal
        ]
    )

    return bimix_gauss.prob(X_plot)



