#######################
# Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px


#######################
# Page configuration
st.set_page_config(
    page_title="FFU DASHBOARD",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded")

#######################
# Load data
@st.cache_data
def get_data_from_csv():
    df = pd.read_csv("data.csv")

    return df

df = get_data_from_csv()

#######################
# Sidebar
with st.sidebar:
    st.title(':bar_chart: FFU DASHBOARD')
    
    # choose spec
    spec_list = list(df.spec.unique())
    selected_spec = st.selectbox('Select a spec', spec_list)
    df_selected_spec = df[df.spec == selected_spec]

    # choose continuous column
    continuous_list = list(df[["power_consumption", "noise", "vibration"]].columns)
    selected_continuous = st.selectbox('Select a continuous variable', continuous_list)
    
    # choose categorical column
    categorical_list = list(df[["motor_type", "3PH/1PH", "filter_type"]].columns)
    selected_categorical = st.selectbox('Select a categorical varible', categorical_list)

#######################
# Plots

# box plot
def make_boxplot():
    boxplot = px.box(df_selected_spec, x="spec", y=selected_continuous)
    return boxplot


def make_barchart():
    # 선택된 범주형 변수의 각 범주 개수 계산
    category_counts = df_selected_spec[selected_categorical].value_counts().reset_index()
    category_counts.columns = [selected_categorical, 'Count']

    # 막대 차트 생성
    barchart = px.bar(category_counts, x=selected_categorical, y='Count')
    return barchart

def make_piechart():
    
    piechart = px.pie(df.spec.value_counts().reset_index(), values="count", names="spec", category_orders={"index": ["a", "b", "c", "d", "e"]})
    return piechart

def make_scatter():
    scatterplot = px.scatter(df, x="air_velocity", y="power_consumption", color="spec")
    return scatterplot


#######################
# Dashboard Main Panel

col = st.columns((1.5, 4.5), gap='medium')


with col[0]:
    st.markdown("### Spec count")
    piechart = make_piechart()
    st.plotly_chart(piechart, use_container_width=True)

    barchart = make_barchart()
    st.plotly_chart(barchart, use_container_width=True)

with col[1]:
    st.markdown('#### Total Population')
    
    boxplot = make_boxplot()
    st.plotly_chart(boxplot, use_container_width=True)

    scatterplot = make_scatter()
    st.plotly_chart(scatterplot, use_container_width=True)



st.markdown("### FFU DATASET")
st.dataframe(df, use_container_width=True)

#######################
# chatbot

from dotenv import load_dotenv
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

user_question = st.text_input("Ask a question about your data")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, 
               model='gpt-3.5-turbo'),        # 모델 정의
    df,                                    # 데이터프레임
    verbose=False,                          # 추론과정 출력
    agent_type=AgentType.OPENAI_FUNCTIONS, # AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if user_question is not None and user_question != "":
    response = agent.invoke(user_question)

    st.write(response)