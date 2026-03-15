from demande.models.demande.adaptive_rff import fit_transform
from demande.configs.adaptive_rff_config import AdaptiveRffParameterConfig

def iterate_dmkde(model, max_epochs, batched_train_data, train_set, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow, 
                  adaptive_rff_parameters:AdaptiveRffParameterConfig, adaptive_activated=None):
    if adaptive_activated is not None and adaptive_activated == True:

        rff_layer = fit_transform(train_set, adaptive_rff_parameters)
        model.fm_x = rff_layer

    model.fit(batched_train_data, epochs=max_epochs, )
    checkpoint.write(file_prefix=checkpoint_path)  # overwrite best val model

