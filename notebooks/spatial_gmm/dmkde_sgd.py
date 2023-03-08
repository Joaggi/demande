try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("2021-2-conditional-density-estimation", "/Doctorado/")

from run_experiment_hyperparameter_search import run_experiment_hyperparameter_search

import sys 

if len(sys.argv) > 1:
    algorithm, database = sys.argv[1], sys.argv[2]
else:
    algorithm, database = "dmkde_sgd", "spatial_gmm"

for j in range(7, 11):
    setting = {
        "z_trainable_layers_0": False,
        "z_trainable_layers_1": True,
        "z_base_lr": 0.001,
        "z_dimension": j,
        "z_max_epochs": 10000,
        "z_adaptive": True,
        "z_server": "remote",
        "z_batch_size": 500,
        "z_adaptive_learning_rate": 0.001,
        "z_adaptive_epochs": 1000,
        "z_adaptive_batch_size": 500,
        "z_random_search_iter": 5,
    }
    #prod_settings = {
    #    "z_dim_rff" :  [ 250, 500, 1000, 1500, 2000], \
    #    "z_dim_eig" :  [ 1, 0.5, 0.2, 0.1], \
    #    "z_initialize_with_rho" :  [ True, False], \
    #    "z_sigma" :  [ 2 ** i for i in range(-20,20)], \
    #}

    prod_settings = {
        "z_dim_rff" :  [ 2000], \
        "z_dim_eig" :  [ 1, 0.5, 0.1], \
        "z_initialize_with_rho" :  [ True, False], \
        "z_sigma" :  [0.05, 0.055, 0.06, 0.065, 0.07, 0.075,  0.08, 0.085, 0.09, 0.095, 0.1, 0.11, 0.12, 0.13] 
    }


    run_experiment_hyperparameter_search(algorithm, database, parent_path, prod_settings, setting)

