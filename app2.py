import streamlit as st
from utils import *
# 상태 저장
if 'page' not in st.session_state:
    st.session_state['page'] = 'HOME'

with st.sidebar:
    if st.button("HOME", type='primary', use_container_width=True): st.session_state['page']='HOME'
    if st.button("CHATBOT"): st.session_state['page']='CHATBOT'
    if st.button("MODEL"): st.session_state['page'] = 'MODEL'


if st.session_state['page']=='HOME': home()
elif st.session_state['page']=='CHATBOT': chatbot()
elif st.session_state['page']=='MODEL': model()
