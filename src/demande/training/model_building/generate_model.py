from demande.training.model_building.generate_model_made import generate_model_made
from demande.training.model_building.generate_model_inverse_maf import generate_model_inverse_maf
from demande.training.model_building.generate_model_planar_flow import generate_model_planar_flow
from demande.training.model_building.generate_model_neural_splines import generate_model_neural_splines
from demande.training.model_building.generate_model_dmkde import generate_model_dmkde
from demande.training.model_building.generate_model_dmkde_sgd import generate_model_dmkde_sgd

from demande.configs.dmkde_sgd_config import DmkdeSgdOptimizerConfig, DmkdeSgdParameterConfig

def generate_model(setting):
    if setting["z_algorithm"] == "made":
        return generate_model_made(setting)
    
    if setting["z_algorithm"] == "inverse_maf":
        return generate_model_inverse_maf(setting)

    if setting["z_algorithm"] == "planar_flow":
        return generate_model_planar_flow(setting)

    if setting["z_algorithm"] == "neural_splines":
        return generate_model_neural_splines(setting)

    if setting["z_algorithm"] == "dmkde":
        return generate_model_dmkde(setting["z_sigma"], setting["z_dimension"], setting["z_dim_rff"], setting["z_random_state"])
        
    if setting["z_algorithm"] == "dmkde_sgd":
        dmkde_sgd_parameters = DmkdeSgdParameterConfig( input_dimension=setting["z_dimension"], sigma=setting["z_sigma"],
                                    eig_dim=setting["z_dim_eig"], rff_dim=setting["z_dim_rff"],
                                    random_state=setting["z_random_state"], layer_0_trainable=setting["z_trainable_layers_0"],
                                                        layer_1_trainable=setting["z_trainable_layers_0"])
        optimizer = DmkdeSgdOptimizerConfig(base_lr=setting["z_base_lr"], decay_steps=setting["z_decay_steps"],
                                            end_lr=setting["z_end_lr"], power=setting["z_power"])


        return generate_model_dmkde_sgd(dmkde_sgd_parameters, optimizer)






