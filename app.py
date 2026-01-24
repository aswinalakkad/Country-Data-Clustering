import streamlit as st
import numpy as np
import pandas as pd
import pickle 

# Load the instances that were created 
with open('final_model.pkl','rb') as file:
    model = pickle.load(file)
    
with open('pca.pkl','rb') as file:
    pca = pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)   

def prediction(input_data):
    
    scaled_data = scaler.transform(input_data)
    pca_data = pca.transform(scaled_data)
    pred = model.predict(pca_data)[0]

    if pred == 0:
        return 'Developing'
    elif pred == 1:
        return 'Developed'
    else:
        return 'Under Developed'

def main():
    st.title('Help International Foundation')
    st.subheader('This application will give the status of the country based on socio-economic factors')

    with st.form("country_form"):
        ch_mort = st.number_input('Child mortality rate', min_value=0.0)
        exp = st.number_input('Exports (% GDP)', min_value=0.0)
        imp = st.number_input('Imports (% GDP)', min_value=0.0)
        hel = st.number_input('Health expenditure (% GDP)', min_value=0.0)
        inc = st.number_input('Average income per person', min_value=0.0)
        inf = st.number_input('Inflation rate')
        life_exp = st.number_input('Life expectancy', min_value=0.0)
        total_fert = st.number_input('Total fertility rate', min_value=0.0)
        gdpp = st.number_input('GDP per capita', min_value=0.0)

        submit = st.form_submit_button("Predict")

    if submit:
        input_list = [[
            ch_mort, exp, imp, hel, inc,
            inf, life_exp, total_fert, gdpp
        ]]
        response = prediction(input_list)
        st.success(response)

# to execute the main function 
if __name__ == '__main__':
    main()
