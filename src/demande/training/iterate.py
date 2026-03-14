import tensorflow as tf
import time
from utils.nf_utils import train_density_estimation, nll
import numpy as np
from demande.training.iterate_flow import iterate_flow
from demande.training.iterate_dmkde import iterate_dmkde
from demande.training.iterate_dmkde_sgd import iterate_dmkde_sgd

def iterate(model, max_epochs, batched_train_data, train_set, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow, setting):
    if algorithm == "dmkde":
        iterate_dmkde(model, max_epochs, batched_train_data, train_set, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow, setting)

    elif algorithm == "dmkde_sgd":
        iterate_dmkde_sgd(model, max_epochs, batched_train_data, train_set, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow, setting)
 
    else:
        iterate_flow(model, max_epochs, batched_train_data, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow)

   

