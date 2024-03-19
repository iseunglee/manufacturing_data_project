#######################
# 대시보드 라이브러리
import streamlit as st
import pandas as pd
import plotly.express as px

#######################
# 페이지 기본 설정 구성
st.set_page_config(
    page_title="FFU DATA DASHBOARD",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto"
)

#######################
# 데이터 불러오기
@st.cache_data # 데이터 캐싱
def get_data_from_csv():
    df = pd.read_csv("./data.csv")
    return df

df = get_data_from_csv()

#######################
# 대시보드
def dashboard():
    with st.sidebar:
        st.title(':bar_chart: FFU DASHBOARD')
        #######################
        # 막대차트 관련 변수
        # 등급 선택
        st.markdown("__막대차트 관련 변수__")
        spec_list = list(df.spec.unique())
        selected_spec = st.selectbox('Select a spec with barchart', spec_list)
        df_selected_spec = df[df.spec == selected_spec]
        # "motor_type", "3PH/1PH", "filter_type" 컬럼 선택
        categorical_list = list(df[["motor_type", "3PH/1PH", "filter_type"]].columns)
        selected_categorical = st.selectbox('Select a categorical varible', categorical_list)

        # 박스플롯 관련 변수
        # 등급 선택
        st.markdown("__이상치 관련 변수__")
        selected_spec_boxplot = st.multiselect('Select specs with boxplot', spec_list, default=spec_list)
        df_selected_spec_boxplot = df[df.spec.isin(selected_spec_boxplot)]
        # "power_consumption", "noise", "vibration" 컬럼 선택
        continuous_list = list(df[["power_consumption", "noise", "vibration"]].columns)
        selected_continuous = st.selectbox('Select a continuous variable', continuous_list)

    #######################
    # 그래프 생성
    # 박스플롯
    def make_boxplot():
        boxplot = px.box(df_selected_spec_boxplot, x="spec", y=selected_continuous)
        return boxplot

    # 막대차트
    def make_barchart():
        # 선택된 범주형 변수의 각 범주 개수 계산
        category_counts = df_selected_spec[selected_categorical].value_counts().reset_index()
        category_counts.columns = [selected_categorical, 'Count']
        # 막대 차트 생성
        barchart = px.bar(category_counts, x=selected_categorical, y='Count')
        return barchart

    # 파이차트
    def make_piechart():
        piechart = px.pie(df.spec.value_counts().reset_index(), values="count", names="spec", category_orders={"index": ["a", "b", "c", "d", "e"]})
        return piechart

    # 산점도
    def make_scatter():
        scatterplot = px.scatter(df, x="air_velocity", y="power_consumption", color="spec")
        return scatterplot
    
    #######################
    # 대시보드 메인 패널
    col = st.columns((0.3, 0.7), gap='large') # 나란히 열로 배치된 다중 요소 컨테이너 2개 삽입
    # 첫 번째 컨테이너
    # 파이차트 + 막대차트
    with col[0]:
        st.markdown("##### 생산 등급 비율")
        piechart = make_piechart()
        st.plotly_chart(piechart, use_container_width=True)

        st.markdown("##### 모터타입, 3상/1상, 필터타입 생산대수")
        barchart = make_barchart()
        st.plotly_chart(barchart, use_container_width=True)
    # 두 번째 컨테이너
    # 박스플롯 + 산점도
    with col[1]:
        st.markdown('##### 전력소비량, 노이즈, 진동 이상치')
        boxplot = make_boxplot()
        st.plotly_chart(boxplot, use_container_width=True)

        st.markdown('##### 전력소비량 대비 풍량')
        scatterplot = make_scatter()
        st.plotly_chart(scatterplot, use_container_width=True)

    st.markdown("##### FFU DATASET")
    st.dataframe(df, use_container_width=True)

dashboard()
