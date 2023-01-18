# import io
# import simplejson
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict
import uvicorn
# import shap
from fastapi import FastAPI, Body, HTTPException, Request, Response
from pydantic import BaseModel

from fastapi.responses import ORJSONResponse

app = FastAPI(default_response_class=ORJSONResponse, title='Loan Default  Prediction', version='1.0',
             description='LighGBMClassifier model is used for prediction')

with open("./model_reduced.pkl", "rb") as f:
    model = pickle.load(f)
    
data = pd.read_csv("reduced_train.csv")

# Creating a class for the attributes input to the ML model.
class Inputs(BaseModel):
    """    
    Class for the input data for new client.
    """

    EXT_SOURCE_1       : float
    EXT_SOURCE_3       : float
    EXT_SOURCE_2       : float
    DAYS_BIRTH         : float
    AMT_GOODS_PRICE    : float
    AMT_CREDIT         : float
    AMT_ANNUITY        : float
    DAYS_EMPLOYED      : float
    CODE_GENDER        : float
    AMT_INCOME_TOTAL   : float
    DAYS_EMPLOYED_PERC : float
    INCOME_CREDIT_PERC : float
    ANNUITY_INCOME_PERC: float
    PAYMENT_RATE       : float

class ID(BaseModel):
    """
    Class for the ID of the client.
    """

    id_client : int

@app.post("/predict_new/")
async def predict_new(input_client: Inputs):
    """
    Predict the probability of target for new client using a trained model.

    Parameters:
        input_client (dict): A dictionary containing the input data for new client.

    Returns:
        float: A float representing the probability of target.
    """
    print(input_client)
    df = pd.DataFrame([input_client.dict().values()], columns=input_client.dict().keys())
    result = model.predict_proba(df)[0][1]
    return result

@app.post("/predict_previous/")
async def predict_previous(id: ID):
    """
    Predict the probability of target for new client using a trained model.

    Parameters:
        input_client (dict): A dictionary containing the input data for new client.

    Returns:
        float: A float representing the probability of target.
    """
    data = pd.read_csv("reduced_train.csv")
    data = data.iloc[[int(id.id_client)]]
    result = model.predict_proba(data)[0][1]
    return result

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Fonction pour l'interpretatbilité du modèle à faire passer dans la partie backend
# ---------------------------------------------------------------------------------
# @app.post("/interpret/")
# async def interpret(sv: List[float], feat_names : List[str]):
#     # Create a SHAP summary plot
#     #plot_html = shap.summary_plot(sv) #, show=False, plot_type="bar")
#     return sv #{"plot_html": plot_html}

# @app.get("/plot_global_shap/")
# def plot():
#     # create a SHAP summary plot
#     plt_img = shap.summary_plot(shap_values)
#     # encode the plot as a PNG image
#     bytes_io = io.BytesIO()
#     plt_img.save(bytes_io, format="PNG")
#     # return the image as a response
#     return Response(bytes_io.getvalue(), media_type="image/png")