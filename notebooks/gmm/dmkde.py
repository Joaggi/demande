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
    algorithm, database = "dmkde", "gmm"


for j in range(2, 11):  
  
    setting = {
        "z_max_epochs": int(1),
        "z_dimension": j
    }

    prod_settings = {
        "z_dim_rff" :  [ 250, 500, 1000, 1500, 2000], \
        "z_sigma" :  [ 2 ** i for i in range(-20,20)], \
    }
    run_experiment_hyperparameter_search(algorithm, database, parent_path, prod_settings, setting)


