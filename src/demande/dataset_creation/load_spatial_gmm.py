import numpy as np    
from sklearn.model_selection import train_test_split 



def load_spatial_gmm(train_size, test_size, dimension):
    print(f"loading spatial_gmm trian_size: {train_size} test_size: {test_size} dimension: {dimension}")
    X_train = np.load(f"data/spatial_gmm/data_{dimension}.npy")
    X_label = np.load(f"data/spatial_gmm/label_{dimension}.npy")
    X_densities = np.load(f"data/spatial_gmm/density_{dimension}.npy")

    X_train, X_test, X_train_densities, X_test_densities, X_train_label, X_test_label = \
        train_test_split(X_train, X_densities, X_label,  train_size=train_size, test_size=test_size, stratify=X_label)

    
    return X_train, X_train_densities, X_test, X_test_densities, X_train_label, X_test_label
