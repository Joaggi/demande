
try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("2021-2-conditional-density-estimation", "/Doctorado/" )
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
datasets = ["arc", "bimodal_l", "binomial", "potential_1", "potential_2", "potential_3",
                "potential_4", "star_eight", "swiss_roll"]

fig, axs = plt.subplots(9, 6, figsize=(80,80))

name_of_experiment =  'conditional-density-experiment'

df = None

for j, dataset in enumerate(datasets):
    for i, algorithm in enumerate(algorithms):

        #mlflow = mlflow_get_experiment(f"tracking_{algorithm}.db", f"registry_{algorithm}.db", name_of_experiment)
        mlflow = mlflow_get_experiment(f"tracking.db", f"registry.db", name_of_experiment)

        client = MlflowClient()
        experiments_list = client.list_experiments()
        print(experiments_list)


        #query = f"params.z_run_name = '{dataset}_{algorithm}' and params.z_step = 'test'"
        query = f"params.z_run_name = '{dataset}_{algorithm}'"



        runs = mlflow.search_runs(experiment_ids="1", filter_string=query, 
            run_view_type=ViewType.ACTIVE_ONLY, output_format="pandas")
        try:
            #runs = runs.groupby(["params.z_algorithm", "params.z_dataset"]).mean()

            runs = runs.sort_values("metrics.spearman_value", ascending=False)

            runs = runs.iloc[0].loc[["params.z_algorithm", "params.z_dataset", "metrics.spearman_value", "metrics.mae_value", "metrics.mae_log_value", "metrics.pearson_value"]]

 
            if df is None:
                df = runs
            else:
                df = pd.concat([df, runs], axis=1)
        except:
            pass

print(df.T.to_csv("conditional-density-estimation.csv"))


