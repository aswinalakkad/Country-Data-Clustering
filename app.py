
import streamlit as st
import numpy as np
import pandas as pd
import pickle 

#Load the instances that were created 

with open('final_model.pkl','rb') as file:
    model = pickle.load(file)
    
with open('pca.pkl','rb') as file:
    pca = pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)   

def prediction(input_data):
    
    scaled_data = scaler.transform(input_data)

    pca_data = pca.transform(scaled_data)

    pred=model.predict(pca_data)[0]

    if pred == 0:
        return 'Developing'
    elif pred == 1:
        return 'Developed'
    else:
        return 'Under Developed'

def main():
    
    st.title('Help International Foundation')
    
    st.subheader('This application will give the statuis of the country based on socio-economic factors')

    ch_mort = st.text_input('Enter the child mortality rate:')

    exp = st.text_input('Enter Exports (% GDP):')

    imp = st.text_input('Enter Imports (% GDP):')
    
    hel = st.text_input('Enter Expentiture on health (% GDP):')
    
    inc = st.text_input('Enter average income per person')
    
    inf =  st.text_input('Enter Inflation rate')
    
    life_exp = st.text_input('Enter Life Expectency')
    
    total_fert = st.text_input('Enter number of children that would be born to each woman')
    
    gdpp = st.text_input('Enter The GDP per capita')

    #save all the input in a 2d list 
    input_list = [[ch_mort,exp,imp,hel,inc,inf,life_exp,total_fert,gdpp]]

    #create a button to predict 
    if st.button('Predict'):
        response = prediction(input_list)
        st.success(response)
#to execute the main function 
if __name__ == '__main__':
    main()
