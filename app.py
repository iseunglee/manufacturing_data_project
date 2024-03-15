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

    #################
    # for prediction model variable
    ml_motertype = st.selectbox("Select a motor type", df.motor_type.unique())
    ml_watt = st.number_input("Input watt")
    ml_PH = st.selectbox("Select a 3PH/1PH", df["3PH/1PH"].unique())
    ml_esp = st.number_input("input esp")
    ml_airvelocity = st.number_input("Input air velocity")
    ml_powerconsumption = st.number_input("Input power consumption")
    ml_powerfactor = st.number_input("Input power factor")
    ml_noise = st.number_input("Input noise")
    ml_filtertype = st.selectbox("Select a filet type", df["filter_type"].unique())
    ml_filterthickness = st.selectbox("Select a filter thickness", df["filter_thickness"].unique())
    ml_filterpressure = st.number_input("Input filter pressure")

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

#######################
# prediction model
import lightgbm as lgb

# 모델 로드 및 학습 데이터 전처리
def load_and_preprocess_data():
    # 학습 데이터 로드
    train_df = pd.read_csv('data.csv')

    # 필요한 열 전처리
    train_df['is_BLDC'] = (train_df['motor_type'] == 'BLDC')
    train_df['is_3PH'] = (train_df['3PH/1PH'] == '3PH')
    train_df['is_ULPA'] = (train_df['filter_type'] == 'ULPA')
    train_df['is_75t'] = (train_df['filter_thickness'] == '75t')

    # 필요한 특성 열 선택
    X_features = train_df[['is_BLDC', 'watt', 'is_3PH', 'esp', 'air_velocity', 'power_consumption',
                        'power_factor', 'noise', 'is_ULPA', 'is_75t', 'filter_pressure']]

    y_tr = train_df['spec']

    return X_features, y_tr

def predict_grade(model, X_features):

    # 사용자 입력값을 데이터프레임으로 생성
    user_data = pd.DataFrame({
        'motor_type': [ml_motertype],
        'watt': [ml_watt],
        '3PH/1PH': [ml_PH],
        'esp': [ml_esp],
        'air_velocity': [ml_airvelocity],
        'power_consumption': [ml_powerconsumption],
        'power_factor': [ml_powerfactor],
        'noise': [ml_noise],
        'filter_type': [ml_filtertype],
        'filter_thickness': [ml_filterthickness],
        'filter_pressure': [ml_filterpressure]
    })

    # 열을 조건에 따라 생성
    user_data['is_BLDC'] = (user_data['motor_type'] == 'BLDC')
    user_data['is_3PH'] = (user_data['3PH/1PH'] == '3PH')
    user_data['is_ULPA'] = (user_data['filter_type'] == 'ULPA')
    user_data['is_75t'] = (user_data['filter_thickness'] == '75t')

    # 입력값을 기존 학습 데이터에 맞춰 전처리
    user_data = user_data[X_features.columns]

    # 등급 예측
    predicted_grade = model.predict(user_data)

    return predicted_grade[0]

X_features, y_tr = load_and_preprocess_data()

# 최적 파라미터로 모델 생성
params = {
        'n_estimators': 724,
        'learning_rate': 0.027515887339946664,
        'max_depth': 7,
        'num_leaves': 36,
        'min_child_samples': 7,
        'min_child_samples': 7,
        'subsample': 0.7996478432659917,
        'colsample_bytree': 0.7993675658472117,
        'random_state': 42,
        'verbose': -1
        }

model = lgb.LGBMClassifier(**params)

# 모델 학습
model.fit(X_features, y_tr)

# 모델 예측
predicted_grade = predict_grade(model, X_features)

# Print input features
st.subheader('Input features')
input_feature = pd.DataFrame([[ml_motertype, ml_watt, ml_PH, ml_esp, ml_airvelocity, ml_powerconsumption, ml_powerfactor, ml_noise, ml_filtertype, ml_filterthickness, ml_filterpressure]],
                            columns=['motor_type', 'watt', '3PH/1PH', 'esp', 'air_velocity', 'power_consumption', 'power_factor', 'noise', 'filter_type', 'filter_thickness', 'filter_pressure'])
st.write(input_feature)

st.subheader('Output')
st.metric('Predicted class', predicted_grade, '')

