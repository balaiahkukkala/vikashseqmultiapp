import pandas as pd
import numpy as np
from joblib import load
import streamlit as st
import pickle
import csv

import mf.mf_prediction_code as mf
import cc.cc_prediction_code as cc
import bp.bp_prediction_code as bp
    
def main():
    st.title("Lite-SeqCNN: A Light-weight Deep CNN Architecture for Protein Function Prediction")
    app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction'])
    if app_mode == "Home":
        st.header("Home")
    elif app_mode == 'Prediction':
        st.header("Model Deployment")
    return app_mode


def convert_df(df):
    dict = {'Predicted_GOs': df}  
       
    df = pd.DataFrame(dict) 
    
   
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def function(y_pred_updated):
    #st.write(y_pred_updated)
    
    csv = convert_df(y_pred_updated)

    values = st.number_input(
            "Pick a number",
            0,len(y_pred_updated))
    st.write("CHOSEN TEST CASE ENTRY:",values)
    if st.button('Print GO IDs'):
        st.write(y_pred_updated[values])
        
    st.download_button(
    label="Download Predicted GOIDs as CSV",
    data=csv,
    file_name='predicted_gos.csv',
    mime='text/csv',
    )
    
def model_prediction():
    
    
    
    choice = st.selectbox(

    'Select the Sub Class you want?',

    ('BP','MF','CC'))



    #displaying the selected option

    st.write('You have selected:', choice)
    if(choice == 'BP'):
        X = load_data()
        if X is not None:
            y_pred = bp.main(X)
            function(y_pred)
    elif(choice =='MF'):
        X = load_data()
        if X is not None:
            y_pred = mf.main(X)
            function(y_pred)
    else:
        X = load_data()
        if X is not None:
            y_pred = cc.main(X)
            function(y_pred)
            
   
def load_data():
    st.markdown("## Dataset :")
    data_file = st.file_uploader("CHOOSE CSV FILE CONTAINING VECTORS TO BE PREDICTED UPON")
    if data_file is not None:
        df = pd.read_csv(data_file, header=None, sep = ',')
        if st.checkbox("Show Data"):
            st.dataframe(df)
        return df
    




    
if __name__ == '__main__':
    choice = main()
    


    
if choice == "Home":
    """Protein Function Prediction"""

   

elif choice == 'Prediction':
    
   
    
    
    
    
    st.markdown("## Model Predictions :")
    
    #if data is not None:
    
    model_prediction()
    
    
    
    
  
    
    
    
    
    
    
    