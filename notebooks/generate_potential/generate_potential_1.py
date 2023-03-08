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


def pot_1(z):
    z_1, z_2 = z[:, 0], z[:, 1]
    norm = tf.sqrt(z_1 ** 2 + z_2 ** 2)
    outer_term_1 = .5 * ((norm - 2) / .4) ** 2
    inner_term_1 = tf.exp((-.5 * ((z_1 - 2) / .6) ** 2))
    inner_term_2 = tf.exp((-.5 * ((z_1 + 2) / .6) ** 2))
    outer_term_2 = tf.math.log(inner_term_1 + inner_term_2 + 1e-7)
    u = outer_term_1 - outer_term_2
    return - u


X = np.loadtxt("data/NF1/NF1_1M.csv")

X_densities = np.exp(pot_1(X).numpy())/6.529

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes = axes.flat
plot_pot_func(pot_1, ax = axes[0]) 
axes[0].set_title('Target density')
axes[1].set_xlim([-4,4])
axes[1].set_ylim([-4,4])
sns.scatterplot(X[:, 0], X[:, 1], c=X_densities, alpha=.2, s = 3 ,ax=axes[1])
axes[1].set_title('Samples')
plt.show()


np.savetxt(fname="data/NF1/NF1_1M_densities.csv", delimiter=" ", X = X_densities)
