import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json
from copy import deepcopy



@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

churn_df_raw = load_data(path = "data/cleaned_telecom_churn_data.csv")
churn_df = deepcopy(churn_df_raw)

st.title("Telecom Churn Prediction")
st.header("Customers Info")

#st.table(data=mpg_df)
if st.checkbox("Show Dataframe"):
    st.subheader("Dataset:")
    st.dataframe(data=churn_df)

left_col, middle_col, right_col = st.columns([3,1,1])


