try:
    from initialization import initialization
except:
    from experiments.initialization import initialization

parent_path = initialization("demande", "../")

from run_experiment_best_configuration import run_experiment_best_configuration

import sys 

if len(sys.argv) > 1:
    algorithm, database = sys.argv[1], sys.argv[2]
else:
    algorithm, database = "inverse_maf", "arc"

run_experiment_best_configuration(algorithm, database, parent_path)

