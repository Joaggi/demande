import numpy as np

def mae_log(true_density, estimated_density):
  result  = 1/true_density.shape[0] * np.sum(np.abs(np.log(true_density) - np.log(estimated_density)))
  return result


