import requests
import streamlit as st
import pickle
#import orjson
import pandas as pd 
import datetime
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
now = datetime.date.today()

# Deployement on local machine
#BACKEND = "http://localhost:8000/"

# Deployement on docker
BACKEND = "http://fastapi:8000/"
 
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_page_config(layout="wide")

with open("./shap_values.pkl", "rb") as f:
    shap_values = pickle.load(f)

with open("./model_reduced.pkl", "rb") as f:
    model = pickle.load(f)

data = pd.read_csv("data_pred.csv")
X_test = pd.read_csv("test_data.csv")

def get_prediction_new(input):
    """
    Makes a request to the specified backend endpoint to retrieve a prediction
    based on the provided input data.

    Parameters:
        input_data (dict): A dictionary containing the input data used to generate the prediction.

    Returns:
        dict: A dictionary containing the prediction results if the request was successful and the response format is json.
    """
    response = requests.post(BACKEND + "predict_new/", json=input)
    return response.json()


def get_prediction_previous(input):
    """
    Makes a request to the specified backend endpoint to retrieve a prediction
    based on the provided input data.

    Parameters:
        input_data (dict): A dictionary containing the input data used to generate the prediction.

    Returns:
        dict: A dictionary containing the prediction results if the request was successful and the response format is json.
    """
    response = requests.post(BACKEND + "predict_previous/", json=input, headers={"Content-Type": "application/json"})
    return response.json()


def get_plot_univarie(data, feature, id_client):
    """
    Generates a univariate plot using seaborn library, highlighting the value of a specific client.

    Parameters:
        data (pd.DataFrame): A pandas DataFrame containing the data to be plotted.
        feature (str): The name of the feature to be plotted.
        id_client (int): The ID of the client to be highlighted in the plot.

    Returns:
        sns.FacetGrid: A seaborn FacetGrid object containing the plot.
    """

    g = sns.FacetGrid(data=data, hue="TARGET", height=5)
    g.map(sns.kdeplot, feature) #, shade=True)
    client = data.at[int(id_client), feature]
    ax = g.axes[0][0]
    ax.axvline(client, c='r')
    return g

def get_linear_gauge(value):
    """
    Generates a linear gauge using plotly library.

    Parameters:
        value (float): The value to be displayed in the gauge.
        title (str): The title of the gauge.

    Returns:
        go.Figure: A plotly Figure object containing the gauge.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': 'Score'},
        gauge = {
                'shape': "bullet",
                'axis': {'range': [None, 1]},
                 'steps' : [
                     {'range': [0, 0.25], 'color': "lightgray"},
                     {'range': [0.25, 0.5], 'color': "gray"},
                     {'range': [0.5, 0.75], 'color': "darkgray"},
                     {'range': [0.75, 1], 'color': "black"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}}))
    return fig


def run():
    ######################################################################################################################
    ##################################################### HOMEPAGE #######################################################
    ######################################################################################################################
    def home_page():
        st.markdown("# Loan Default Prediction 💸")
        st.subheader("This machine learning app will help you to make a prediction to help you with your decision!")
        st.write(data.shape)
        st.write("""To borrow money, credit analysis is performed. Credit analysis involves the measure to investigate
                    the probability of the applicant to pay back the loan on time and predict its default/ failure to pay back.""")

    ######################################################################################################################
    ################################################## New Client Page ###################################################
    ######################################################################################################################
    # Previous client prediction
    def new_client():
        st.markdown("# To predict default/failure to pay back status, you need to follow the steps below:")
        st.markdown("""
        1. Enter/choose the client's parameters that best descibe your applicant on the left side bar;
        2. Press the "Predict" button and wait for the result.""")
        
        st.sidebar.markdown("# New client prediction")

        # User input for new client
        EXT_SOURCE_1      = st.sidebar.slider("Enter Normalized score n°1 :", min_value=0., max_value=1., value=0.5, step=0.01)
        EXT_SOURCE_2      = st.sidebar.slider("Enter Normalized score n°2 :", min_value=0., max_value=1., value=0.5, step=0.01)
        EXT_SOURCE_3      = st.sidebar.slider("Enter Normalized score n°3 :", min_value=0., max_value=1., value=0.5, step=0.01)
        st.sidebar.caption("Normalized score from external data source. Value between 0 and 1.")
        DAYS_BIRTH        = -(now - st.sidebar.date_input("Enter client date of birth :", value=datetime.date(1980,1,1), min_value = datetime.date(1970,1,1), max_value = datetime.date.today())).days
        st.sidebar.caption("Client's age in days at the time of application")
        st.write(DAYS_BIRTH)
        AMT_GOODS_PRICE   = st.sidebar.text_input("Enter the price of the goods", value="20000")
        st.sidebar.caption("For consumer loans it is the price of the goods for which the loan is given.")
        AMT_CREDIT        = st.sidebar.text_input("Enter the amount of the credit", value="1000")
        st.sidebar.caption("Credit amount of the loan. Max authorized is 5+06 euros.")
        AMT_ANNUITY       = st.sidebar.text_input("Loan annuity", value="120")
        st.sidebar.caption("Loan annuity in month.")
        DAYS_EMPLOYED     = -(now - st.sidebar.date_input("Days employed :", value=datetime.date(2000,1,1), min_value = datetime.date(1980,1,1), max_value = datetime.date.today())).days
        st.sidebar.caption("How many days before the application the person started current employment.")
        CODE_GENDER       =  st.sidebar.text_input("Gender", value="1")
        st.sidebar.caption("Gender of the client. 1=F, 0=M")
        AMT_INCOME_TOTAL  = st.sidebar.text_input("Total Income", value="10000")
        st.sidebar.caption("Income of the client.")

        # Feature engineering
        AMT_ANNUITY = float(AMT_ANNUITY)*30.43 # 30,43 is the conversion in days
        DAYS_EMPLOYED_PERC  = DAYS_EMPLOYED / DAYS_BIRTH
        INCOME_CREDIT_PERC  = float(AMT_INCOME_TOTAL) / float(AMT_CREDIT)
        ANNUITY_INCOME_PERC = float(AMT_ANNUITY) / float(AMT_INCOME_TOTAL)
        PAYMENT_RATE        = float(AMT_ANNUITY) / float(AMT_CREDIT)

        # Create input dictionary
        input_dict = {
        'EXT_SOURCE_1'       : float(EXT_SOURCE_1),
        'EXT_SOURCE_3'       : float(EXT_SOURCE_3),
        'EXT_SOURCE_2'       : float(EXT_SOURCE_2),
        'DAYS_BIRTH'         : float(DAYS_BIRTH),
        'AMT_GOODS_PRICE'    : float(AMT_GOODS_PRICE),
        'AMT_CREDIT'         : float(AMT_CREDIT),
        'AMT_ANNUITY'        : float(AMT_ANNUITY),
        'DAYS_EMPLOYED'      : float(DAYS_EMPLOYED),
        'CODE_GENDER'        : float(CODE_GENDER),
        'AMT_INCOME_TOTAL'   : float(AMT_INCOME_TOTAL),
        'DAYS_EMPLOYED_PERC' : float(DAYS_EMPLOYED_PERC),
        'INCOME_CREDIT_PERC' : float(INCOME_CREDIT_PERC),
        'ANNUITY_INCOME_PERC': float(ANNUITY_INCOME_PERC),
        'PAYMENT_RATE'       : float(PAYMENT_RATE)
        }
    

        btn_predict = st.sidebar.button("Predict")
        btn_explain = st.sidebar.button("Explain")

        # Prediction
        if btn_predict:
            with st.spinner("Waiting for the prediction..."):
                prediction = get_prediction_new(input_dict)
                #st.success(f"The prediction from model: {prediction}")
                st.write("The prediction from model: ", prediction)

                fig = get_linear_gauge(prediction)  
                st.plotly_chart(fig)
                    
        # Explanation
        if btn_explain:
                placeholder1 = st.empty()
                with placeholder1.container(): 
                    st.subheader('Result Interpretability - Applicant Level')
                    df = pd.DataFrame([input_dict])
                    shap.initjs()
                    explainer = shap.Explainer(model[-1], X_test)
                    shap_values_ = explainer(df)
                    fig = shap.plots.bar(shap_values_)
                    st.pyplot(fig)
                    st.write(""" In this chart blue and red mean the feature value. """)
                    st.write(""" The size of the bar means the importance of the feature. """)

    ######################################################################################################################
    ############################################# Previous Client Page ###################################################
    ######################################################################################################################
    # Previous client prediction
    def previous_client():
        st.markdown("# Previous client prediction analysis")
        st.sidebar.markdown("# Previous client prediction")
        id_client = st.sidebar.text_input("Enter client ID", value="1")
        st.sidebar.caption("Enter ID client between 0 and 307511 include please.")

        input_id = {'id_client': id_client}
        btn_predict = st.sidebar.button("Predict")
        btn_explain = st.sidebar.button("Explain")

        # Prediction
        if btn_predict:
            with st.spinner("Waiting for the prediction..."):
                prediction = get_prediction_previous(input_id)
                st.write("The prediction from model: ", prediction)
                    
        
        placeholder3 = st.empty()
        with placeholder3.container():
            st.subheader('Result Interpretability - Applicant Level')
            # Explanation
            if btn_explain:
                with st.spinner("Waiting for the explanation..."):
                    col1, col2 = st.columns(2, gap="large")
                    
                    with col1:
                        st.header("Global")
                        shap.initjs()
                        fig_g = shap.summary_plot(shap_values)
                        st.pyplot(fig_g)


                    with col2:
                        st.header("Local")
                        shap.initjs()
                        plotg = shap.waterfall_plot(shap_values[0])
                        st.pyplot(plotg)
        

        features_lst = ['PAYMENT_RATE', 'EXT_SOURCE_1', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'DAYS_BIRTH']
        st.write(features_lst)

        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            #st.markdown("### Second Chart")
            feature_1 = st.selectbox('Select feature 1', features_lst)
            plt_feat1 = get_plot_univarie(data, feature_1, int(id_client))
            st.pyplot(plt_feat1)

            
        with fig_col2:
            feature_2 = st.selectbox('Select feature 2', features_lst)
            plt_feat2 = get_plot_univarie(data, feature_2, int(id_client))
            st.pyplot(plt_feat2)
        
        #placeholder2 = st.empty()
        #with placeholder2:
        st.markdown("### Bivariate chart")
        features = st.multiselect('What are your favorite colors',['PAYMENT_RATE', 'EXT_SOURCE_1', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'DAYS_BIRTH'],['EXT_SOURCE_2', 'DAYS_BIRTH'])
        st.write(features[0])
        fig = px.scatter(data, x=features[0], y=features[1], color='TARGET', opacity=0.3)
                        
        st.plotly_chart(fig, use_container_width=True)


    page_names_to_funcs = {
        "Home page": home_page,
        "New Client": new_client,
        "Previous client": previous_client,
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == '__main__':
   #by default it will run at 8501 port
   run()