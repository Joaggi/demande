import matplotlib.pyplot as plt
import tensorflow as tf

def plot_graph_density(true_density, estimated_density, path_name):
  params = {
   'axes.labelsize': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [7.0, 6.0]
   }
  plt.rcParams.update(params)
  plt.figure()
  f, ax = plt.subplots(figsize=(6, 6))


  ax.scatter(estimated_density , true_density, s=3)
  #ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
  maximum_density = tf.reduce_max(estimated_density) if tf.reduce_max(estimated_density) > tf.reduce_max(true_density) else tf.reduce_max(true_density)
  ident = [0.0, maximum_density]
  plt.plot(ident,ident, color="red") 
  plt.xlabel("Estimated density")
  plt.ylabel("True density")
  plt.savefig(path_name)
  #plt.show()


