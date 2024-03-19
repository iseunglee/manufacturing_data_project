#######################
# 예측 모델 라이브러리
import streamlit as st
import lightgbm as lgb
import pandas as pd

#######################
# 페이지 기본 설정 구성
st.set_page_config(
    page_title="FFU SPEC PREDICTION",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

#######################
# 데이터 불러오기
@st.cache_data
def get_data_from_csv():
    df = pd.read_csv("./data.csv")
    return df

df = get_data_from_csv()

#######################
# 예측 모델   
def model():
    with st.sidebar:
        # 모델에 입력할 변수
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
    
    # 데이터 불러오기 및 전처리 함수
    def load_and_preprocess_data():
        # 학습 데이터 로드
        train_df = pd.read_csv('./data.csv')
        # 필요한 열 전처리
        train_df['is_BLDC'] = (train_df['motor_type'] == 'BLDC')
        train_df['is_3PH'] = (train_df['3PH/1PH'] == '3PH')
        train_df['is_ULPA'] = (train_df['filter_type'] == 'ULPA')
        train_df['is_75t'] = (train_df['filter_thickness'] == '75t')
        # 독립변수 선택
        X_features = train_df[['is_BLDC', 'watt', 'is_3PH', 'esp', 'air_velocity', 'power_consumption',
                            'power_factor', 'noise', 'is_ULPA', 'is_75t', 'filter_pressure']]
        # 종속변수 선택
        y_tr = train_df['spec']
        return X_features, y_tr

    # 등급 예측 함수
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
        # 사용자 입력값 전처리
        user_data['is_BLDC'] = (user_data['motor_type'] == 'BLDC')
        user_data['is_3PH'] = (user_data['3PH/1PH'] == '3PH')
        user_data['is_ULPA'] = (user_data['filter_type'] == 'ULPA')
        user_data['is_75t'] = (user_data['filter_thickness'] == '75t')
        # 입력값을 기존 학습 데이터에 맞춰 전처리
        user_data = user_data[X_features.columns]
        # 등급 예측
        predicted_grade = model.predict(user_data)
        return predicted_grade[0]

    # 전처리된 데이터 생성
    X_features, y_tr = load_and_preprocess_data()

    # 최적 파라미터
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

    # 최적 파라미터로 모델 생성
    model = lgb.LGBMClassifier(**params)

    # 모델 학습
    model.fit(X_features, y_tr)

    # 모델 예측
    predicted_grade = predict_grade(model, X_features)

    # 메인 패널에 사용자 입력 변수 값 띄우기
    st.subheader('Input features')
    input_feature = pd.DataFrame([[ml_motertype, ml_watt, ml_PH, ml_esp, ml_airvelocity, ml_powerconsumption, ml_powerfactor, ml_noise, ml_filtertype, ml_filterthickness, ml_filterpressure]],
                                columns=['motor_type', 'watt', '3PH/1PH', 'esp', 'air_velocity', 'power_consumption', 'power_factor', 'noise', 'filter_type', 'filter_thickness', 'filter_pressure'])
    st.write(input_feature)
    
    # Predict 버튼 눌렀을 때 예측값 출력
    if st.button("Predict"):
        st.subheader('Output')
        st.metric('Predicted class', predicted_grade, '')

model()