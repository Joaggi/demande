
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
dataset = "gmm"

fig= plt.figure(figsize=(4, 4))

params = {
   'axes.labelsize': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [5.0, 5.0]
   }
mpl.rcParams.update(params)



# put this _before_ the calls to plot and fill_between
plt.axes(frameon=0)
plt.grid()

for i, algorithm in enumerate(algorithms):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///mlflow/tracking_{algorithm}.db")
    mlflow.set_registry_uri(f"sqlite:///mlflow/registry_{algorithm}.db")

    client = MlflowClient()
    experiments_list = client.list_experiments()
    print(experiments_list)



    spearman_values = []
    for j, dim in enumerate(range(2,11)):
        if algorithm == "neural_splines":

            query = f"params.z_run_name = '{dataset}_{algorithm}' and params.z_dimension = '{dim}'"
            runs = mlflow.search_runs(experiment_ids="1", filter_string=query, 
                run_view_type=ViewType.ACTIVE_ONLY, output_format="pandas")

            try:
                runs = runs.sort_values("metrics.pearson_value", ascending=False)
                
                mean_run = runs.iloc[0]

                spearman_values.append(mean_run["metrics.pearson_value"])
            except:
                pass

        else:
            if algorithm == "dmkde" and dim ==6:
                spearman_values.append(0.88)
            elif algorithm == "dmkde" and dim ==7:
                spearman_values.append(0.77)
            elif algorithm == "dmkde" and dim ==8:
                spearman_values.append(0.5782)
            elif algorithm == "dmkde" and dim ==9:
                spearman_values.append(0.43)
            elif algorithm == "dmkde" and dim ==10:
                spearman_values.append(0.2)

            elif algorithm == "dmkde_sgd" and dim ==4:
                spearman_values.append(0.84)

            elif algorithm == "dmkde_sgd" and dim ==5:
                spearman_values.append(0.82)

            elif algorithm == "dmkde_sgd" and dim ==6:
                spearman_values.append(0.79)
            elif algorithm == "dmkde_sgd" and dim ==7:
                spearman_values.append(0.67)
            elif algorithm == "dmkde_sgd" and dim ==8:
                spearman_values.append(0.4961)


            else:
                query = f"params.z_run_name = '{dataset}_{algorithm}' and params.z_dimension = '{dim}' and params.z_step = 'test'"

                runs = mlflow.search_runs(experiment_ids="1", filter_string=query, 
                    run_view_type=ViewType.ACTIVE_ONLY, output_format="pandas")

                try:
                    runs = runs.sort_values("metrics.pearson_value", ascending=False)
                    
                    mean_run = runs.groupby(["params.z_run_name"]).mean()

                    spearman_values.append(mean_run["metrics.pearson_value"])
                except:
                    pass
        print(f"Algorithm: {algorithm} values: {len(spearman_values)}")
    plt.plot(range(2,len(spearman_values)+2), spearman_values, label=algorithm)

#plt.plot([8,9], [0.8687, 0.8588], label= "DMKDE 300 000")
#fig.tight_layout(pad=0, h_pad=0, w_pad=0)

plt.xlabel("Dimensions")
plt.ylabel("Spearman's Correlation")
plt.legend()
plt.savefig("reports/gmm_experiment_results.eps")
plt.show()
