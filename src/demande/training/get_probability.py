
from demande.models.demande.calculate_constant_dmkde import calculate_constant_dmkde

def get_probability(model, X, algorithm, dimension=None, sigma=None):
    if algorithm in ["dmkde", "dmkde_sgd"]:
        assert dimension!=None and sigma!=None

        gamma= 1/ (2*sigma**2)
        return calculate_constant_dmkde(gamma, dimension=dimension) * model.predict(X)
        
    else:
        return model.prob(X).numpy()

