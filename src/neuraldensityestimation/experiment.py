from generate_model import generate_model 
from iterate import iterate
from plot_probability import plot_probability
from calculate_metrics import calculate_metrics
from plot_graph_density import plot_graph_density
from define_grid import define_grid
from load_dataset import load_dataset
from get_current_day import get_current_day
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from get_probability import get_probability
import mlflow_wrapper

def experiment( train_size, test_size, setting, mlflow, parent_dir):
    dimension = setting["z_dimension"]
    batch_size = setting["z_batch_size"]
    dataset = setting["z_dataset"]
    print(setting)

    active_run = mlflow_wrapper.start_run(mlflow, run_name=setting["z_run_name"])
    string_date = get_current_day()

    from pathlib import Path
    checkpoint_prefix = parent_dir + "/mlflow/tensorflow/" + setting["z_run_name"] + "/" + string_date + "/tf_ckpt/"
    Path(checkpoint_prefix).mkdir(parents=True, exist_ok=True)


    X_train, X_train_density, X_test, X_test_densities = load_dataset(dataset, train_size, test_size, dimension)


    train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    batched_train_data = train_dataset.batch(batch_size)
    print(f'##----------#----------------------#---------------#')
    print(f'Train 2D data set shape is:{X_train.shape} ')
    print(f'Test 2D data set shape is:{X_test.shape} and their densities shape is {X_test_densities.shape}')
    print(f'##----------#----------------------#---------------#')

    plt.figure()
    plt.axes(frameon = 0)
    plt.grid()
    plt.scatter(X_test[:,0], X_test[:,1], c = X_test_densities, alpha=.3,s = 3, linewidths= 0.0000001)
    plt.colorbar()
    plt.savefig(f"reports/original_data_2d.png")
    #plt.show()
    mlflow_wrapper.log_artifact(mlflow, f"reports/original_data_2d.png")

    model = generate_model(setting)

    polinomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(setting["z_base_lr"], \
            setting["z_decay_steps"], setting["z_end_lr"], power=setting["z_power"])
    opt = tf.keras.optimizers.Adam(learning_rate=polinomial_decay)  # optimizer

    checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)

    iterate(model, setting["z_max_epochs"], batched_train_data, X_train, X_test, opt, checkpoint, checkpoint_prefix, setting["z_algorithm"], mlflow, setting)

    if("verbose" in setting and setting["verbose"]): f"Iterate finished {i}"

    if X_train.shape[1] == 2:
        x_grid, y_grid, xy_plot_grid, x_number, y_number = define_grid(X_train)
        probability = get_probability(model, xy_plot_grid, setting)
        print(int(x_number), int(y_number))
        plot_probability(probability, x_grid, y_grid, round(x_number), round(y_number), path_name=f'reports/grid_mesh_probability.png')
        mlflow_wrapper.log_artifact(mlflow, f'reports/grid_mesh_probability.png')

    estimated_density = get_probability(model, X_test, setting)

    metrics = calculate_metrics(X_test_densities, estimated_density)

    count = 0
    while(count < 10):
        try:
            save_mlflow_experiment(mlflow, setting, metrics, parent_dir, X_test_densities, string_date, estimated_density)
            break
        except Exception as e:
            logging.critical(e, exc_info=True)
            print("Error at saving")
            try:
                mlflow_wrapper.end_run(mlflow)
                active_run = mlflow_wrapper.start_run(mlflow, run_name=setting["z_run_name"])
            except:
                pass

        count = count + 1

def save_mlflow_experiment(mlflow, setting, metrics, parent_dir, X_test_densities, string_date, estimated_density):
    mlflow_wrapper.log_params(mlflow, setting)
    mlflow_wrapper.log_metrics(mlflow, metrics)
    mlflow_wrapper.log_artifact(mlflow, parent_dir + "/mlflow/tensorflow/" + setting["z_run_name"] + "/" + string_date)
    if setting["z_plot_graph_density"]:
        plot_graph_density(X_test_densities, estimated_density, path_name=f'reports/test_density.png')
        mlflow_wrapper.log_artifact(mlflow, f'reports/test_density.png')

    mlflow_wrapper.end_run(mlflow)


