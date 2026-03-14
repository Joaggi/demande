def run_experiment_best_configuration(algorithm, database, parent_path, custom_setting = None):
    from mlflow_get_experiment import mlflow_get_experiment 
    from generate_product_dict import generate_product_dict, add_random_state_to_dict
    from experiment import experiment
    from get_best_run import get_best_run
    from convert_best_hyperparameter_search_settings import convert_best_hyperparameter_search_settings
    from generate_several_dict_with_random_state import generate_several_dict_with_random_state
    import logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    import datetime

    # %% [markdown]
    # Experiments
    #%% #import tensorflow as tf 
    #tf.keras.mixed_precision.set_global_policy("mixed_float16")


    experiments_folder = "/content/drive/My Drive/Academico/doctorado_programacion/"

    today = datetime.date.today().strftime('%Y%m%d')
    
    print("#--------------------------------#-----------#")
    setting = {
        "z_base_lr": 1e-3,
        "z_end_lr": 1e-7,
        "z_max_epochs": int(25),
        "z_decay_steps": int(1000),
        "z_train_size": 40000,
        "z_test_size": 1000,
        "z_weight_decay": 1e-4, 
        "z_batch_size": 64,
        "z_power": 0.5,
        "z_algorithm": algorithm,
        "z_experiment": f"{database}_{algorithm}",
        "z_dataset": f"{database}",
        "experiments_folder": experiments_folder,
        "logs": experiments_folder + "logs/",
        "models": experiments_folder + "models/",
        "datasets": experiments_folder + "datasets/",
        "sufix_name_experiment" : f"{database}_{algorithm}",
        "z_dimension": 2,
        "z_run_name": f"{database}_{algorithm}",
        "z_plot_graph_density": True,
        "z_name_of_experiment": 'conditional-density-experiment',
        #"z_random_search": True,
        "z_random_search": False,
        #"z_random_search_random_state": 1,
        #"z_random_search_iter": 30,
        "z_test_running_times": 1,
        "z_step": "test",
        "z_date": today,
        "z_adaptive_epochs": 1000
    }
 
    if custom_setting is not None:
        setting = dict(setting, **custom_setting)
    
    # %% [markdown]
    # Inicializar experimentación
    
    #%%
    # %% [markdown]
    # Create mlflow experiment 

    mlflow = mlflow_get_experiment(f"tracking_{setting['z_algorithm']}.db", f"registry_{setting['z_algorithm']}.db", setting["z_name_of_experiment"])

    best_experiment = get_best_run(mlflow, setting, "metrics.pearson_value")

    best_experiment = convert_best_hyperparameter_search_settings(best_experiment)

    best_experiment = dict(best_experiment, **setting)

    settings_test = generate_several_dict_with_random_state(best_experiment, setting["z_test_running_times"])

    #mlflow = mlflow_get_experiment(f"tracking.db", f"registry.db", setting["z_name_of_experiment"], server="remote")
    mlflow = mlflow_get_experiment(f"tracking.db", f"registry.db", setting["z_name_of_experiment"])

    for i, setting in enumerate(settings_test):
        if("verbose" in setting and setting["verbose"]): print(i) 
        experiment(train_size = setting["z_train_size"], test_size = setting["z_test_size"], 
            setting = setting, mlflow = mlflow, parent_dir = parent_path)


