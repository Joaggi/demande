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
    algorithm, database = "made", "gmm"


for j in range(2, 11):
    setting = {
        "z_dimension": j
    }


    prod_settings = {
            "z_hidden_shape" :  [[20,20], [50, 50], [100, 100], [200,200], [500,500], [1000, 1000]], \
            "z_n_layers" : [2,4,8,12,24], \
            "z_base_lr" :  [1e-3, 1e-4, 1e-5]
    }


    run_experiment_hyperparameter_search(algorithm, database, parent_path, prod_settings, setting)

