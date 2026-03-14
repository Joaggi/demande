try:
    from initialization import initialization
except:
    from experiments.initialization import initialization

parent_path = initialization("demande", "/Doctorado/")

from run_experiment_best_configuration import run_experiment_best_configuration

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

    run_experiment_best_configuration(algorithm, database, parent_path, setting)

