
try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("2021-2-conditional-density-estimation", "/Doctorado/" )
from mlflow_get_experiment import mlflow_get_experiment 
from generate_product_dict import generate_product_dict, add_random_state_to_dict
from generate_model import  generate_model
from experiment import experiment
from convert_best_hyperparameter_search_settings import convert_best_hyperparameter_search_settings
from transform_params_to_settings import transform_params_to_settings
from define_grid import define_grid
from get_probability import get_probability
from plot_probability import plot_probability
from load_dataset import load_dataset
from calculate_probability import calculate_probability


import tensorflow as tf
import mlflow

name_of_experiment = 'conditional-density-experiment'

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
import os
algorithms = ["real", "dmkde", "dmkde_sgd", "made", "inverse_maf", "planar_flow", "neural_splines"]
algorithms_display = ["Real", "Dmkde", "Dmkde Sgd", "Made", "Inverse Maf", "Planar Flow", "Neural Splines"]

#algorithms = ["real"]
#algorithms_display = ["real"]

datasets = ["arc", "bimodal_l", "binomial", "potential_1", "potential_2", "potential_3",
                "potential_4", "star_eight", "swiss_roll"]
datasets_display = ["Arc", "Bimodal l", "Binomial", "Potential 1", "Potential 2", "Potential 3",
                "Potential 4", "Star Eight", "Swiss Roll"]


fig, axs = plt.subplots(9, 7, figsize=(45,60))
fig2, axs2 = plt.subplots(9, 6, figsize=(45,60))

for j, dataset in enumerate(datasets):

    #X_train, X_train_density, X_test, X_test_densities = load_dataset(setting["z_dataset"], setting["z_train_size"], setting["z_test_size"], setting["z_dimension"])
    X_train, X_train_density, X_test, X_test_densities = load_dataset(dataset, 2000, 1000, 2)
    
    plt.axes(frameon = 0)
    
    x_grid, y_grid, xy_plot_grid, x_number, y_number = define_grid(X_train)

    for i, algorithm in enumerate(algorithms):
        if algorithm == "real":
            probability = calculate_probability(xy_plot_grid, dataset)
            axs[j,i].contourf(x_grid, y_grid, probability.reshape([round(x_number), round(y_number)]))
            axs[j,i].axis("off")

        else:
            #mlflow = mlflow_get_experiment(f"tracking_{algorithm}.db", f"registry_{algorithm}.db", name_of_experiment)
            mlflow = mlflow_get_experiment(f"tracking.db", f"registry.db", name_of_experiment)

            if algorithm in [ "made", "inverse_maf", "planar_flow", "neural_splines"]:
                query = f"params.z_run_name = '{dataset}_{algorithm}' and params.z_step = 'test' and params.z_test_running_times = '1'"
            elif algorithm in ["dmkde"]:
                query = f"params.z_run_name = '{dataset}_{algorithm}' and params.z_step = 'test'"
            elif algorithm in ["dmkde_sgd"]:
                query = f"params.z_run_name = '{dataset}_{algorithm}' and params.z_experiment = '2'"



            runs = mlflow.search_runs(experiment_ids="1", filter_string=query, 
                run_view_type=ViewType.ACTIVE_ONLY, output_format="pandas")

            try:
                runs = runs.sort_values("metrics.spearman_value", ascending=False)


                best_run = runs.iloc[0]

                setting = transform_params_to_settings(best_run)

                setting = convert_best_hyperparameter_search_settings(setting)

                client = MlflowClient()
                experiments_list = client.list_experiments()
                print(experiments_list)



                files = client.list_artifacts(best_run.run_id)

                local_dir = "/tmp/artifact_downloads"
                if not os.path.exists(local_dir):
                    os.mkdir(local_dir)
                local_path = client.download_artifacts(best_run.run_id, "", local_dir)
                print("Artifacts downloaded in: {}".format(local_path))
                print("Artifacts: {}".format(os.listdir(local_path)))

                setting["z_algorithm"] = algorithm
                model = generate_model(setting)

                path_checkpoint = files[0].path

                path_full_checkpoint = f"{local_path}{path_checkpoint}/"


                polinomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(setting["z_base_lr"], \
                        setting["z_decay_steps"], setting["z_end_lr"], power=setting["z_power"])
                optimizer = tf.keras.optimizers.Adam(learning_rate=polinomial_decay)  # optimizer


                try:

                    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                    if len(os.listdir(path_full_checkpoint)) > 1:
                        print("model")

                    checkpoint.restore(path_full_checkpoint + "tf_ckpt/")


                    print("Model restored")

                    probability = get_probability(model, xy_plot_grid, setting)
                    print(int(x_number), int(y_number))
                    axs[j,i].contourf(x_grid, y_grid, probability.reshape([round(x_number), round(y_number)]))
                    axs[j,i].axis("off")

                    estimated_density = get_probability(model, X_test, setting)

                    axs2[j,i-1].scatter(estimated_density , X_test_densities, s=3)
                    #axs2[j,i].axis("off")
                    #ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
                    maximum_density = tf.reduce_max(estimated_density) if tf.reduce_max(estimated_density) > tf.reduce_max(X_test_densities) else tf.reduce_max(X_test_densities)
                    ident = [0.0, maximum_density]
                    axs2[j,i-1].plot(ident,ident, color="red") 
                    #plt.xlabel("Estimated density")
                    #plt.ylabel("True density")

                except Exception as error:
                    print("Cannot restore the model ")
                    print( error)



                #img = plt.imread(f"{local_dir}/test_density.png")
                #img = plt.imread(f"{local_dir}/grid_mesh_probability.png")

                #axs[j,i].imshow(img)
                #axs[j,i].axis("off")
                #axs[j,i].set_title(algorithm)
            except Exception as error:
                print(error)



for ax, col in zip(axs[0], algorithms_display):
    ax.set_title(col)

for ax, row in zip(axs[:,0], datasets_display):
    ax.set_ylabel(row, rotation=0, size='large')

fig.savefig("/tmp/image_density.eps", bbox_inches="tight", pad_inches=0)

fig2.savefig("/tmp/image_comparison_spearman.eps", bbox_inches="tight", pad_inches=0)
#plt.show()
