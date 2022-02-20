#!/usr/bin/env python
# coding: utf-8

# In[18]:


import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML


# In[19]:


h2o.init()


# In[20]:


automl = h2o.upload_mojo('StackedEnsemble_AllModels_4_AutoML_1_20220219_03122.zip')

st.title("Veeva Telecom Test Prediction App")


file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])


if file_upload is not None:
    data = pd.read_csv(file_upload)
    data = h2o.H2OFrame(data)
    predictions = automl.predict(data).as_data_frame()
    #predictions = automl.predict(data).as_data_frame()
    predictions = predictions.join(data.as_data_frame())
    st.write(predictions)
    
    @st.cache
    
    def convert_df(df):
        return df.to_csv(index = False, header=True).encode('utf-8')
    csv = convert_df(predictions)
    
    st.download_button(label="Download data as CSV",data=csv,
                file_name='credit_risk_prediction.csv',mime='text/csv')
    

        
            
            
            


# In[21]:


# FOR SINGLE PREDICTIONS, TRY ST.WRITE INSTEAD OF ST.SUCCESS


# In[ ]:




