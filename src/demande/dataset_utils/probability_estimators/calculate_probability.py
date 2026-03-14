#from calculate_probability_gmm import calculate_probability_gmm
#from calculate_probability_spatial_gmm import calculate_probability_spatial_gmm
from calculate_probability_potential_1 import calculate_probability_potential_1
from calculate_probability_potential_2 import calculate_probability_potential_2
from calculate_probability_potential_3 import calculate_probability_potential_3
from calculate_probability_potential_4 import calculate_probability_potential_4
from calculate_probability_arc import calculate_probability_arc
from calculate_probability_swiss_roll import calculate_probability_swiss_roll
from calculate_probability_star_eight import calculate_probability_star_eight
from calculate_probability_bimodal_l import calculate_probability_bimodal_l
from calculate_probability_binomial import calculate_probability_binomial


def calculate_probability(X, dataset_name):
    if dataset_name == "gmm":
        return calculate_probability_gmm(X)

    elif dataset_name == "spatial_gmm":
        X_train, X_train_densities, X_test, X_test_densities, _, _= calculate_probability_spatial_gmm(X)
        return X_train, X_train_densities, X_test, X_test_densities

    elif dataset_name == "potential_1":
        return calculate_probability_potential_1(X).numpy()

    elif dataset_name == "potential_2":
        return calculate_probability_potential_2(X).numpy()

    elif dataset_name == "potential_3":
        return calculate_probability_potential_3(X).numpy()

    elif dataset_name == "potential_4":
        return calculate_probability_potential_4(X).numpy()

    elif dataset_name == "arc":
        return calculate_probability_arc(X).numpy()

    elif dataset_name == "swiss_roll":
        return calculate_probability_swiss_roll(X)

    elif dataset_name == "star_eight":
        return calculate_probability_star_eight(X).numpy()

    elif dataset_name == "bimodal_l":
        return calculate_probability_bimodal_l(X).numpy()

    elif dataset_name == "binomial":
        return calculate_probability_binomial(X).numpy()





