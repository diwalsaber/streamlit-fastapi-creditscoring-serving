import requests
import streamlit as st 
import pandas as pd 
import datetime
import streamlit.components.v1 as components
now = datetime.date.today()


def run():
    backend = "http://fastapi:8000/predict"
    st.title("Loan Prediction")

    PAYMENT_RATE            = st.sidebar.slider("Please choose your payement rate amount:",min_value=0., max_value=0.5,step=0.01)
    EXT_SOURCE_1            = st.sidebar.slider("Please choose your EXT_SOURCE_1 :", min_value=0., max_value=1.,step=0.01)
    EXT_SOURCE_2            = st.sidebar.slider("Please choose your EXT_SOURCE_2 :", min_value=0., max_value=1.,step=0.01)
    EXT_SOURCE_3            = st.sidebar.slider("Please choose your EXT_SOURCE_3 :", min_value=0., max_value=1.,step=0.01)
    DAYS_BIRTH              = (now - st.sidebar.date_input("Please choose your day of birth", value=None, min_value = datetime.date(1970,1,1), max_value = datetime.date.today())).days


    input_dict = {'PAYMENT_RATE': PAYMENT_RATE,
    'EXT_SOURCE_1': EXT_SOURCE_1,
    'EXT_SOURCE_3': EXT_SOURCE_3,
    'EXT_SOURCE_2': EXT_SOURCE_2,
    'DAYS_BIRTH': DAYS_BIRTH,
    }

    btn_predict = st.sidebar.button("Predict")

    if btn_predict:
        
        response = requests.post(backend, json=input_dict)
        prediction = response.text
        st.success(f"The prediction from model: {prediction}")

if __name__ == '__main__':
    #by default it will run at 8501 port
    run()