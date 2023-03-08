try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("2021-2-conditional-density-estimation", "/Doctorado/")

from run_experiment_best_configuration import run_experiment_best_configuration

import sys 

if len(sys.argv) > 1:
    algorithm, database = sys.argv[1], sys.argv[2]
else:
    algorithm, database = "neural_splines", "gmm"

for j in range(10, 11):
    setting = { 
        "z_train_size": 300000,
        "z_test_size": 1000,
        "z_dimension": j,
        "z_max_epochs": 5,
    }


    run_experiment_best_configuration(algorithm, database, parent_path, setting)

