from demande.dataset_creation.load_gmm import load_gmm
from demande.dataset_creation.load_spatial_gmm import load_spatial_gmm
from demande.dataset_creation.load_potential_1 import load_potential_1
from demande.dataset_creation.load_potential_2 import load_potential_2
from demande.dataset_creation.load_potential_3 import load_potential_3
from demande.dataset_creation.load_potential_4 import load_potential_4
from demande.dataset_creation.load_arc import load_arc
from demande.dataset_creation.load_swiss_roll import load_swiss_roll
from demande.dataset_creation.load_star_eight import load_star_eight
from demande.dataset_creation.load_bimodal_l import load_bimodal_l
from demande.dataset_creation.load_binomial import load_binomial


def load_dataset(dataset, train_size, test_size, dimension):
    if(dataset == "gmm"):
        return load_gmm(train_size, test_size, dimension)

    if(dataset == "spatial_gmm"):
        X_train, X_train_densities, X_test, X_test_densities, _, _= load_spatial_gmm(train_size, test_size, dimension)
        return X_train, X_train_densities, X_test, X_test_densities

    if(dataset == "potential_1"):
        return load_potential_1(train_size, test_size, dimension)

    if(dataset == "potential_2"):
        return load_potential_2(train_size, test_size, dimension)

    if(dataset == "potential_3"):
        return load_potential_3(train_size, test_size, dimension)

    if(dataset == "potential_4"):
        return load_potential_4(train_size, test_size, dimension)

    if(dataset == "arc"):
        return load_arc(train_size, test_size, dimension)

    if(dataset == "swiss_roll"):
        return load_swiss_roll(train_size, test_size, dimension)

    if(dataset == "star_eight"):
        return load_star_eight(train_size, test_size, dimension)

    if(dataset == "bimodal_l"):
        return load_bimodal_l(train_size, test_size, dimension)

    if(dataset == "binomial"):
        return load_binomial(train_size, test_size, dimension)








