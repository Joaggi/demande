import mlflow_wrapper
import mlflow
import os

def mlflow_create_experiment(tracking_uri, registry_uri, name_of_experiment, server="local"):

    if server == "local":
        mlflow_wrapper.set_tracking_uri(mlflow, "sqlite:///mlflow/" + tracking_uri)
        mlflow_wrapper.set_registry_uri(mlflow, "sqlite:///mlflow/" + registry_uri)
        try:
          mlflow_wrapper.create_experiment(mlflow, name_of_experiment, "mlflow/")
        except:
          print("Experiment already created")
        mlflow_wrapper.set_experiment(mlflow, name_of_experiment)

    else:

        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        artifact_uri = os.environ["MLFLOW_ARTIFACT_URI"]

        mlflow_wrapper.set_tracking_uri(mlflow, tracking_uri)
        mlflow_wrapper.set_registry_uri(mlflow, tracking_uri)
        try:
          mlflow_wrapper.create_experiment(mlflow, name_of_experiment, artifact_uri)
        except:
          print("Experiment already created")
        mlflow_wrapper.set_experiment(mlflow, name_of_experiment)


    return mlflow

