#######################
# 챗봇 라이브러리
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType

#######################
# 페이지 기본 설정 구성
st.set_page_config(
    page_title="FFU DATA CHATBOT",
    page_icon="💬",
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
# 챗봇
def chatbot():
    # api key 불러오기
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input(placeholder="내용을 입력하세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo-0613",
            openai_api_key=API_KEY,
            streaming=True
        )
        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            agent_type=AgentType.OPENAI_FUNCTIONS,   
        )
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

st.markdown("## FFU 생산 데이터 챗봇")
chatbot()

