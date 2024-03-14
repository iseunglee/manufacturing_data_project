import plotly_express as px
import streamlit as st
import pandas as pd

st.set_page_config(page_title="FFU DASHBORAD",
                   page_icon=":bar_chart:",
                   layout="wide")

# 데이터 로드
@st.cache_data
def get_data_from_csv():
    df = pd.read_csv("data.csv")

    return df

df = get_data_from_csv()

# -------SIDEBAR--------

st.sidebar.header("Please Filter Here:")

spec = st.sidebar.multiselect(
    "Select the Spec:",
    options=df["spec"].unique(),
    default=df["spec"].unique()
)

col = st.sidebar.selectbox(
    "Select variation that you want to check",
    options=df[["power_consumption", "noise", "vibration"]].columns
)

df_selection = df.query(
    "spec == @spec & col == @col"
)

# --------MAINPAGE--------
st.title(":bar_chart: FFU DASHBOARD")
st.markdown("##")

st.dataframe(df_selection, use_container_width=True)

fig = px.box(df_selection, x= "spec", y="options")

st.plotly_chart(fig, use_container_width=True)

