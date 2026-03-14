from get_best_val_experiment import get_best_val_experiment
import mlflow_wrapper

def get_best_run(mlflow, setting, metric_to_evaluate):
    dataset = setting['z_dataset']
    algorithm = setting['z_algorithm']
    dimension = setting['z_dimension']
    query = f"params.z_run_name = '{dataset}_{algorithm}' and params.z_dimension = '{dimension}'"

    experiments_list = mlflow_wrapper.get_experiment_by_name(mlflow, setting["z_name_of_experiment"])
    experiment_id = experiments_list.experiment_id

    return get_best_val_experiment(mlflow, experiment_id,  query, metric_to_evaluate)

