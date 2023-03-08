import numpy as np

def mae(true_density, estimated_density):
  result  = 1/true_density.shape[0] * np.sum(np.abs(true_density - estimated_density))
  return result


