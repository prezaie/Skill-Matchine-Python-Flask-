from model import knn

def prediction(model,X_tr, X_te, X_tr_sc, X_te_sc):
    """ Prediction closest employees to our query position
    Args:
        model (Class): DataModel created from our training data
        X_tr (DataFrame): Train Dataframe
        X_te (DataFrame): Test Dataframe
        X_tr_sc (DataFrame): Standardized Train DataFrame
        X_te_sc (DataFrame): Standardized Test DataFrame    
    Returns:
        X_tr.loc[m1,m2] (DataFrame): Query results
    """
    
    poi = X_te_sc
    closest_points = model.kneighbors(poi.reshape(1, -1), 3, return_distance=False)
    # for i,close_id in enumerate(closest_points[0]):
    #     idx = X_tr.index[close_id]
    #     m1 = (X_tr.index == idx)
    #     m2 = (X_tr[m1] != 0).all()
    #return X_tr.loc[m1,m2]  
    
    return X_tr.loc[closest_points[0],:]
    