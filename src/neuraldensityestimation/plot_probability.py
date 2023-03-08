import matplotlib.pyplot as plt

def plot_probability(probability, x, y, size_x, size_y, path_name="probability.pdf"):
  plt.axes(frameon = 0)
  params = {
    'axes.labelsize': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [7.0, 6.0]
    }
  plt.rcParams.update(params)

  print("log max probability:", probability)
  print("log min probability:", probability)

  #fig2 = plt.figure()
  #ax2 = fig2.add_subplot(111)
  plt.contourf(x, y, probability.reshape([size_x, size_y]))
  plt.colorbar()
  plt.savefig(path_name)
  #plt.show()

