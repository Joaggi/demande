import tensorflow as tf
import time
from utils.train_utils import train_density_estimation, nll
import numpy as np
from adaptive_rff import fit_transform


def iterate_dmkde(model, max_epochs, batched_train_data, train_set, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow, setting):
    if setting["z_adaptive"] is not None and setting["z_adaptive"] == True:
        rff_layer = fit_transform(setting, train_set)
        model.fm_x = rff_layer

    model.fit(batched_train_data, epochs=max_epochs, )
    checkpoint.write(file_prefix=checkpoint_path)  # overwrite best val model

