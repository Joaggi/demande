from demande.training.model_building.generate_model_dmkde import generate_model_dmkde
from demande.training.iterate_dmkde import iterate_dmkde

from demande.models.demande.adaptive_rff import fit_transform

from demande.configs.dmkde_sgd_config import DmkdeSgdParameterConfig
from demande.configs.adaptive_rff_config import AdaptiveRffParameterConfig

def iterate_dmkde_sgd(model, max_epochs, batched_train_data, train_set, test_set, opt, checkpoint, 
                      checkpoint_path, algorithm, mlflow, adaptive_rff_parameter: AdaptiveRffParameterConfig, 
                      dmkde_sgd_parameter: DmkdeSgdParameterConfig):

    if dmkde_sgd_parameter.adaptive_activated is not None and dmkde_sgd_parameter.adaptive_activated == True:
        rff_layer = fit_transform(train_set, adaptive_rff_parameter)
        model.fm_x = rff_layer
        model.fm_x.trainable = False

   
    if dmkde_sgd_parameter.initialize_with_rho:
        dmkde = generate_model_dmkde(dmkde_sgd_parameter.sigma, dmkde_sgd_parameter.input_dimension, dmkde_sgd_parameter.rff_dim,
                                     dmkde_sgd_parameter.random_state)
        iterate_dmkde(dmkde, 1, batched_train_data, train_set, test_set, opt, checkpoint, checkpoint_path, algorithm, mlflow, adaptive_rff_parameter, dmkde_sgd_parameter.adaptive_activated)
        if dmkde_sgd_parameter.adaptive_activated:
            eig_vals = model.set_rho(dmkde.weights[0])
        else:
            eig_vals = model.set_rho(dmkde.weights[2])

    model.fit(batched_train_data, epochs=max_epochs )
    
    checkpoint.write(file_prefix=checkpoint_path)  # overwrite best val model

