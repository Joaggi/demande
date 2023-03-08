import json 
import logging

def convert_best_hyperparameter_search_settings(best_experiment: dict) -> dict:
    """
        Function designed to obtain params from experiments when 
        the returned values from mlflow are not a primitive data but a string

        The library json was used to acomplished that task.
    """

    best_experiment = dict(best_experiment, **{"z_step": "test"})
    # for param in params_int:
    #     best_experiment[param] = int(best_experiment[param])
    # for param in params_float:
    #     best_experiment[param] = float(best_experiment[param])

    # if params_boolean is not None:
    #     for param in params_boolean:
    #         best_experiment[param] = bool(best_experiment[param])
        
    for param in best_experiment.keys():
        try:
            best_experiment[param] = json.loads(best_experiment[param])
        except:
            logging.debug(f"Param: {param} is a string")
            pass
    return best_experiment


