#######################
# chatbot libraries
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler
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

    #################################
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="What is this data about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)



        llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo", openai_api_key=API_KEY, streaming=True
        )

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)



st.title("랭체인")
chatbot()

