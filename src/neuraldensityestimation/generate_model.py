from generate_model_made import generate_model_made
from generate_model_inverse_maf import generate_model_inverse_maf
from generate_model_planar_flow import generate_model_planar_flow
from generate_model_neural_splines import generate_model_neural_splines
from generate_model_dmkde import generate_model_dmkde
from generate_model_dmkde_sgd import generate_model_dmkde_sgd


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
        return generate_model_dmkde(setting)
        
    if setting["z_algorithm"] == "dmkde_sgd":
        return generate_model_dmkde_sgd(setting)






