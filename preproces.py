from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess(df, X_te):
    """Standardize data

    Args:
        df (str, or DataFrame): input data
        test_size (float) : size of test set 
    Returns:
        X_tr (Dataframe): Train DataFrame
        X_te (Dataframe): Test DataFrame
        X_tr_sc (Numpy array): Standardized Train DataFrame
        X_te_sc (Numpy array): Standardized Test DataFrame

    """   
    df = df.astype(float) 
    X_te = X_te.astype(float)
    sc = StandardScaler()
    X_tr = df.drop(columns='project_employee_id')
    X_te = X_te.drop(columns='project_employee_id')
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
      
    return X_tr, X_te, X_tr_sc, X_te_sc