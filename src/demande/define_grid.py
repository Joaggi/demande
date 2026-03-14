import numpy as np
import math 

def define_grid(X):

  x_min, y_min = X.min(axis=0)
  x_min, y_min = math.floor(x_min), math.floor(y_min)
  x_max, y_max = X.max(axis=0)
  x_max, y_max = math.ceil(x_max), math.ceil(y_max)

  x_grid = (x_min,x_max,0.1)
  x_number = (x_grid[1] - x_grid[0])/x_grid[2]
  y_grid = (y_min,y_max,0.1)
  y_number = (y_grid[1] - y_grid[0])/y_grid[2]


  
  x, y = np.mgrid[x_grid[0]:x_grid[1]:x_grid[2], y_grid[0]:y_grid[1]:y_grid[2]]
  pos = np.dstack((x, y))
  X_plot = pos.reshape([int(round(x_number)*round(y_number)),2])
  X_plot
  return x, y, X_plot, x_number, y_number


