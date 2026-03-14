import numpy as np    



def load_gmm(train_size, test_size, dimension):
    print(f"loading gmm trian_size: {train_size} test_size: {test_size} dimension: {dimension}")
    X_train = np.load("data/gmm/data_ten_dimensions.npy")[:train_size, :dimension]
    X_test = np.load("data/gmm/data_test_ten_dimensions.npy")[:test_size, :dimension]
    X_test_densities = np.load("data/gmm/probability_test_ten_dimensions_" + str(dimension) + ".npy")[:test_size]
    
    return X_train, None, X_test, X_test_densities
