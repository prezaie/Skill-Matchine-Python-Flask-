from read import data_reader
from preproces import preprocess
from model import knn
import pandas as pd
import numpy as np
from prediction import prediction
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json

"""Connects to Flask to get the search query. Using KNN to return most similar results for skill match
    Args:
        json data
    Returns:
        df (Dataframe): Dataframe contains best matches employee IDs and skills
"""

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods = ['POST'])
def predict():
        skills= request.json['skill']
        headers = [str(i) for i in skills.split(',')]
        payload= request.json['data']
        values = [float(i) for i in payload.split(',')]
        df=data_reader()
        col_list=list(df.columns)
        df_new=pd.DataFrame(columns=col_list)
        X=pd.DataFrame([values],columns=headers, 
                                dtype=float)                    
        X_te=pd.concat([df_new,X], axis=0,sort=False)
        X_te.fillna(0,inplace=True) 
                           
        X_tr, X_te, X_tr_sc, X_te_sc = preprocess(df,X_te)
        my_model = knn(X_tr_sc, 5, 0.4)
        prediction_results=prediction(my_model,X_tr, X_te, X_tr_sc, X_te_sc)
        prediction_results=prediction_results.loc[:, (prediction_results != 0).any(axis=0)]
        return json.dumps(json.loads(prediction_results.to_json(orient='index')))
if __name__ == '__main__':
    app.run(port=6000, debug=True)
