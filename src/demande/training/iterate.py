import tensorflow as tf
import time
from utils.nf_utils import train_density_estimation, nll
import numpy as np
from demande.training.iterate_flow import iterate_flow
from demande.training.iterate_dmkde import iterate_dmkde
from demande.training.iterate_dmkde_sgd import iterate_dmkde_sgd
from demande.configs.adaptive_rff_config import AdaptiveRffParameterConfig
from demande.configs.dmkde_sgd_config import DmkdeSgdParameterConfig

def iterate(model, max_epochs, batched_train_data, train_set, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow, setting):
    if algorithm == "dmkde":
        adaptive_rff_parameters = AdaptiveRffParameterConfig(
                input_dimension=setting["z_dimension"],
                learning_rate=setting["z_adaptive_learning_rate"], batch_size=setting["z_adaptive_batch_size"], epochs=setting["z_adaptive_epochs"],
                 sigma=setting["z_sigma"], rff_dim=setting["z_dim_rff"],
        ) 
        iterate_dmkde(model, max_epochs, batched_train_data, train_set, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow, 
                      adaptive_rff_parameters, setting["z_adaptive"])

    elif algorithm == "dmkde_sgd":
        adaptive_rff_parameters = AdaptiveRffParameterConfig(
                input_dimension=setting["z_dimension"],
                learning_rate=setting["z_adaptive_learning_rate"], batch_size=setting["z_adaptive_batch_size"], epochs=setting["z_adaptive_epochs"],
                 sigma=setting["z_sigma"], rff_dim=setting["z_dim_rff"],
        )
        dmkde_sgd_parameters = DmkdeSgdParameterConfig( input_dimension=setting["z_dimension"], sigma=setting["z_sigma"],
                                eig_dim=setting["z_dim_eig"], rff_dim=setting["z_dim_rff"],
                                random_state=setting["z_random_state"], layer_0_trainable=setting["z_trainable_layers_0"],
                                                    layer_1_trainable=setting["z_trainable_layers_0"])

        iterate_dmkde_sgd(model, max_epochs, batched_train_data, train_set, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow, 
                          adaptive_rff_parameters, dmkde_sgd_parameters)
 
    else:
        iterate_flow(model, max_epochs, batched_train_data, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow)

   

