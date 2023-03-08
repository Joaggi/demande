
try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("2021-2-conditional-density-estimation", "/Doctorado/", 
        "/content/drive/", "My Drive/Academico/doctorado_programacion/experiments/")
from mlflow_create_experiment import mlflow_create_experiment 
from generate_product_dict import generate_product_dict, add_random_state_to_dict
from experiment import experiment

import pandas as pd

import mlflow


from mlflow_get_experiment import mlflow_get_experiment 

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
import os

algorithms = ["dmkde", "dmkde_sgd", "made", "inverse_maf", "planar_flow", "neural_splines"]
datasets = ["gmm"]

fig, axs = plt.subplots(9, 6, figsize=(80,80))

name_of_experiment =  'conditional-density-experiment'

df = None

for j, dataset in enumerate(datasets):
    for i, algorithm in enumerate(algorithms):
        for dimension in  range(2, 11):

            mlflow = mlflow_get_experiment(f"tracking_{algorithm}.db", f"registry_{algorithm}.db", name_of_experiment)

            client = MlflowClient()
            experiments_list = client.list_experiments()
            print(experiments_list)



            query = f"params.z_run_name = '{dataset}_{algorithm}' and params.z_step = 'test' and params.z_dimension = '{dimension}'"



            runs = mlflow.search_runs(experiment_ids="1", filter_string=query, 
                run_view_type=ViewType.ACTIVE_ONLY, output_format="pandas")
            try:
                runs = runs.groupby(["params.z_algorithm", "params.z_dataset", "params.z_dimension"]).mean()

                if df is None:
                    df = runs
                else:
                    df = pd.concat([df, runs])
            except:
                pass

print(df.to_csv("conditional-density-estimation-gmm.csv"))
