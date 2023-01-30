import pickle
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI(title='Loan Default Prediction', version='1.0',
             description='LighGBMClassifier model is used for prediction')

with open("./model_saved.pkl", "rb") as f:
    model = pickle.load(f)

# Charger l'objet "explainer"
#with open('explainer.pkl', 'rb') as file:
#    explainer = pickle.load(file)    

# Charge les valeurs SHAP à partir du fichier
#with open("shap_values.pkl", "rb") as f:
#    shap_values = pickle.load(f)

data = pd.read_csv("reduced_train.csv")
#X_test = pd.read_csv("X_test.csv")
#X_train = pd.read_csv("X_train.csv")
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


@app.get('/')
@app.get('/home')

def read_home():
    """
    Home endpoint which can be used to test the availability of the application.

    """
    return {'message': 'System is healthy'}

@app.post("/predict_new")
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

@app.post("/predict_previous")
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

# @app.get("/explain")
# def explain(request: Request):
#     #shap_values = request.json()
#     #X_train = request.json()
#     #explainer = shap.Explainer(model[-1].predict_proba, X_train)
#     #shap_values = explainer(X_test)
#     #return {"expected_value": explainer.expected_value, "shap_values": shap_values.tolist()}
#     return shap_values
#     #return {"shap_values": shap_values.tolist()}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

