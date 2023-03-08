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
    algorithm, database = "dmkde_sgd", "arc"

setting = {
    "z_trainable_layers_0": False,
    "z_trainable_layers_1": True,
    "z_experiment": 2,
    "z_adaptive": True,
    "z_max_epochs": int(500),
    "z_random_search_iter": 20,
}

#prod_settings = {
#    "z_dim_rff" :  [ 1000, 2000], \
#    "z_dim_eig" :  [ 1, 0.1], \
#    "z_initialize_with_rho" :  [ True, False], \
#    "z_sigma" :  [  i for i in [0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 1, 1.2, 1.4, 1.6]], \
#}

prod_settings = {
    "z_dim_rff" :  [ 1000], \
    "z_dim_eig" :  [ 1, 0.1], \
    #"z_initialize_with_rho" :  [ True, False], \
    "z_initialize_with_rho" :  [ False], \
    "z_sigma" :  [  i for i in [0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2]], \
}



run_experiment_hyperparameter_search(algorithm, database, parent_path, prod_settings, setting)

