#######################
# chatbot libraries
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType

#######################
# Load data
@st.cache_data
def get_data_from_csv():
    df = pd.read_csv("../data.csv")

    return df

df = get_data_from_csv()

#######################
# chatbot
def chatbot():
    
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")

    user_question = st.text_input("Ask a question about your data")

    agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model='gpt-3.5-turbo'), df, verbose=False, agent_type=AgentType.OPENAI_FUNCTIONS)

    if user_question is not None and user_question != "":
        response = agent.invoke(user_question)
        st.write(response)

chatbot()
