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
    algorithm, database = "dmkde_sgd", "gmm"

for j in range(2, 11):
    setting = {
        "z_trainable_layers_0": False,
        "z_trainable_layers_1": True,
        "z_base_lr": 1e-4,
        "z_dimension": j,
        "z_max_epochs": 200
    }

    prod_settings = {
        "z_dim_rff" :  [ 16000], \
        "z_dim_eig" :  [ 0.05], \
        #"z_initialize_with_rho" :  [ True, False], \
        "z_initialize_with_rho" :  [ False], \
        "z_sigma" :  [ 0.8 + i * 0.2 for i in range(10)], \
    }



    run_experiment_hyperparameter_search(algorithm, database, parent_path, prod_settings, setting)

