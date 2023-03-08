import tensorflow as tf
import time
from utils.train_utils import train_density_estimation, nll
import numpy as np

from generate_model_dmkde import generate_model_dmkde
from iterate_dmkde import iterate_dmkde

from adaptive_rff import fit_transform

def iterate_dmkde_sgd(model, max_epochs, batched_train_data, train_set, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow, setting):
    if setting["z_adaptive"] is not None and setting["z_adaptive"] == True:
        rff_layer = fit_transform(setting, train_set)
        model.fm_x = rff_layer
        model.fm_x.trainable = False

   
    if setting["z_initialize_with_rho"]:
        setting_dmkde = setting
        setting_dmkde["z_algorithm"] = "dmkde"
        dmkde = generate_model_dmkde(setting_dmkde)
        iterate_dmkde(dmkde, 1, batched_train_data, train_set, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow, setting)
        if setting["z_adaptive"]:
            eig_vals = model.set_rho(dmkde.weights[0])
        else:
            eig_vals = model.set_rho(dmkde.weights[2])

    model.fit(batched_train_data, epochs=max_epochs )
    
    checkpoint.write(file_prefix=checkpoint_path)  # overwrite best val model

