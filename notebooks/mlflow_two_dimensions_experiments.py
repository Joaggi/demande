
try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("2021-2-conditional-density-estimation", "/Doctorado/" )
from mlflow_create_experiment import mlflow_create_experiment 
from generate_product_dict import generate_product_dict, add_random_state_to_dict
from experiment import experiment

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow/tracking.db")
mlflow.set_registry_uri("sqlite:///mlflow/registry.db")

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
import os

client = MlflowClient()
experiments_list = client.list_experiments()
print(experiments_list)


algorithms = ["dmkde", "dmkde_sgd", "made", "inverse_maf", "planar_flow", "neural_splines"]
algorithms_display = ["Dmkde", "Dmkde Sgd", "Made", "Inverse Maf", "Planar Flow", "Neural Splines"]
datasets = ["arc", "bimodal_l", "binomial", "potential_1", "potential_2", "potential_3",
                "potential_4", "star_eight", "swiss_roll"]
datasets_display = ["Arc", "Bimodal l", "Binomial", "Potential 1", "Potential 2", "Potential 3",
                "Potential 4", "Star Eight", "Swiss Roll"]


fig, axs = plt.subplots(9, 6, figsize=(45,60))

for j, dataset in enumerate(datasets):
    for i, algorithm in enumerate(algorithms):
        query = f"params.z_run_name = '{dataset}_{algorithm}'"



        runs = mlflow.search_runs(experiment_ids="1", filter_string=query, 
            run_view_type=ViewType.ACTIVE_ONLY, output_format="pandas")

        try:
            runs = runs.sort_values("metrics.spearman_value", ascending=False)

            best_run = runs.iloc[0]


            client.list_artifacts(best_run.run_id)

            local_dir = "/tmp/artifact_downloads"
            if not os.path.exists(local_dir):
                os.mkdir(local_dir)
            local_path = client.download_artifacts(best_run.run_id, "", local_dir)
            print("Artifacts downloaded in: {}".format(local_path))
            print("Artifacts: {}".format(os.listdir(local_path)))


            img = plt.imread(f"{local_dir}/test_density.png")
            #img = plt.imread(f"{local_dir}/grid_mesh_probability.png")

            axs[j,i].imshow(img)
            axs[j,i].axis("off")
            #axs[j,i].set_title(algorithm)
        except:
            pass



for ax, col in zip(axs[0], algorithms_display):
    ax.set_title(col)

for ax, row in zip(axs[:,0], datasets_display):
    ax.set_ylabel(row, rotation=0, size='large')

plt.axis("off")
plt.savefig("/tmp/imagen_2.eps", bbox_inches="tight", pad_inches=0)
#plt.show()
