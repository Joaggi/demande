
from calculate_constant_dmkde import calculate_constant_dmkde

def get_probability(model, X, setting):
    if setting['z_algorithm'] in ["dmkde", "dmkde_sgd"]:
        
        sigma = setting["z_sigma"]
        gamma= 1/ (2*sigma**2)

        return calculate_constant_dmkde(gamma, dimension=setting['z_dimension']) * model.predict(X)
        
    else:
        return model.prob(X).numpy()

