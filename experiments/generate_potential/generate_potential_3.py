import numpy as np
import tensorflow as tf 
import matplotlib.pylab as plt
import seaborn as sns

def plot_pot_func(pot_func, ax=None):
    if ax is None:
        _, ax = plt.subplots(1)
    x = np.linspace(-4, 4, int(1e2))
    y = np.linspace(-4, 4, int(1e2))
    xx, yy = np.meshgrid(x, y)
    in_tens = tf.constant(np.vstack([xx.ravel(), yy.ravel()]).T)
    z = (tf.exp(pot_func(in_tens))).numpy().reshape(xx.shape)

    cmap = plt.get_cmap('inferno')
    ax.contourf(x, y, z.reshape(xx.shape), cmap=cmap)

def w_1(z):
    return tf.sin((2 * np.pi * z[:, 0]) / 4)

def w_2(z):
    return 3 * tf.exp(-.5 * ((z[:, 0] - 1) / .6) ** 2)

def pot_3(z):
    term_1 = tf.exp(-.5 * (
        (z[:, 1] - w_1(z)) / .35) ** 2)
    term_2 = tf.exp(-.5 * (
        (z[:, 1] - w_1(z) + w_2(z)) / .35) ** 2)
    u = - tf.math.log(term_1 + term_2 + 1e-7)
    return tf.exp(- u)/ 13.934



X = np.load("data/NF3/nf3.npy")

X_densities = pot_3(X).numpy()

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes = axes.flat
plot_pot_func(pot_3, ax = axes[0]) 
axes[0].set_title('Target density')
axes[1].set_xlim([-4,4])
axes[1].set_ylim([-4,4])
sns.scatterplot(X[:, 0], X[:, 1], c=X_densities, alpha=.2, s = 3 ,ax=axes[1])
axes[1].set_title('Samples')
plt.show()


np.savetxt(fname="data/NF3/NF3_densities.csv", delimiter=" ", X = X_densities)
