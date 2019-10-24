from sklearn.neighbors import NearestNeighbors

def knn(X_tr_sc, n_neighbors, radius):
    
    """ Unsupervised learner for implementing neighbor searches

    Args:
        X_tr_sc (DataFrame): Standardized Train DataFrame
        n_neighbors (int): Number of neighbors
        radius (float): Range of parameter space
    Returns:
        model (Class): DataModel that fitted to the training data
     
    """    

    neigh = NearestNeighbors(n_neighbors, radius,metric='euclidean')
    model = neigh.fit(X_tr_sc) 
    
    return model