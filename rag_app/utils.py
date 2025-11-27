import streamlit as st

def init_session():
    if "chat_history_ui" not in st.session_state:
        st.session_state.chat_history_ui = []

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
