import streamlit as st


# Here I am defining a function to initialize session state variables for chat history and QA chain in the Streamlit app.
def init_session():
    if "chat_history_ui" not in st.session_state:
        st.session_state.chat_history_ui = []

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
