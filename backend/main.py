import json
import pickle
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title='Car Price Prediction', version='1.0',
             description='Linear Regression model is used for prediction')

with open("./model_saved.pkl", "rb") as f:
    model = pickle.load(f)

# Creating a class for the attributes input to the ML model.
class Inputs(BaseModel):
	PAYMENT_RATE : float
	EXT_SOURCE_1 : float
	EXT_SOURCE_2 : float
	EXT_SOURCE_3 : float
	DAYS_BIRTH   : float

@app.get('/')
@app.get('/home')

def read_home():
    """
    Home endpoint which can be used to test the availability of the application.

    """
    return {'message': 'System is healthy'}

@app.post("/predict")
def predict(inputs: Inputs):

    df = pd.DataFrame(columns=['PAYMENT_RATE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH'],
                      data=np.array([inputs.PAYMENT_RATE,inputs.EXT_SOURCE_1,inputs.EXT_SOURCE_2,
                      inputs.EXT_SOURCE_3,inputs.DAYS_BIRTH]).reshape(1,5))
    
    result = model.predict_proba(df)

    return result

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

