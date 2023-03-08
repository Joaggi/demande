
try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("2021-2-conditional-density-estimation", "/Doctorado/")
from mlflow_create_experiment import mlflow_create_experiment 
from generate_product_dict import generate_product_dict, add_random_state_to_dict
from experiment import experiment


from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
import os
import matplotlib as mpl

algorithms = ["dmkde", "dmkde_sgd", "made", "inverse_maf", "planar_flow", "neural_splines"]
dataset = "spatial_gmm"


params = {
   'axes.labelsize': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [30.0, 30.0]
   }
mpl.rcParams.update(params)

fig= plt.figure(figsize=(8, 6))


# put this _before_ the calls to plot and fill_between
plt.axes(frameon=0)
plt.grid()

tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
artifact_uri = os.environ["MLFLOW_ARTIFACT_URI"]

import mlflow
mlflow.set_tracking_uri( tracking_uri)
mlflow.set_registry_uri( tracking_uri)

client = MlflowClient()
experiment_name = "conditional-density-experiment"
print(client.list_experiments())
experiment_list = client.get_experiment_by_name(experiment_name)
experiment_id = experiment_list.experiment_id



for i, algorithm in enumerate(algorithms):

    

    spearman_values = []
    for j, dim in enumerate(range(2,11)):
        #query = f"params.z_run_name = '{dataset}_{algorithm}' and params.z_dimension = '{dim}' and params.z_step = 'test'"
        query = f"params.z_run_name = '{dataset}_{algorithm}' and params.z_dimension = '{dim}'"

        runs = mlflow.search_runs(experiment_ids=experiment_id, filter_string=query, 
            run_view_type=ViewType.ACTIVE_ONLY, output_format="pandas")

        try:
            runs = runs.sort_values("metrics.spearman_value", ascending=False)
            
            spearman_values.append(runs.iloc[0]["metrics.spearman_value"])
            #mean_run = runs.groupby(["params.z_run_name"]).mean()

            #spearman_values.append(mean_run["metrics.pearson_value"])
        except:
            pass
        print(f"Algorithm: {algorithm} values: {len(spearman_values)}")
    plt.plot(range(2,len(spearman_values)+2), spearman_values, label=algorithm)

#plt.plot([8,9], [0.8687, 0.8588], label= "DMKDE 300 000")
#fig.tight_layout(pad=0, h_pad=0, w_pad=0)

plt.xlabel("Dimensions")
plt.ylabel("Spearman's Correlation")
plt.legend(loc="lower left")

fig.savefig("/tmp/gmm_spatial_experiment.eps", bbox_inches="tight", pad_inches=0)
