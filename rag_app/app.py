import streamlit as st
from loaders import load_document
from rag_chain import create_rag_chain
from utils import init_session
# -----------------------------------
# Streamlit Init
# -----------------------------------
st.set_page_config(page_title="RAG Chat With Memory", page_icon="💬")
st.title("📚 RAG Chat App (DeepSeek + ChromaDB + Memory)")

init_session()

# -----------------------------------
# Clear Chat
# -----------------------------------
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history_ui = []
    st.session_state.qa_chain = None
    st.success("Chat cleared. Upload file again.")

# -----------------------------------
# File Upload
# -----------------------------------
uploaded_file = st.file_uploader("Upload PDF / TXT / DOCX")

if uploaded_file:
    docs = load_document(uploaded_file)
    print("docs_file_path :", docs)

    st.session_state.qa_chain = create_rag_chain(docs)
    st.success("File processed. Start chatting!")

# -----------------------------------
# Chat Input
# -----------------------------------
user_input = st.chat_input("Ask something...")

if user_input and st.session_state.qa_chain:
    st.session_state.chat_history_ui.append(("user", user_input))

    with st.spinner("Thinking..."):
        response = st.session_state.qa_chain({"question": user_input})

    answer = response["answer"]
    st.session_state.chat_history_ui.append(("assistant", answer))

# -----------------------------------
# Display Chat
# -----------------------------------
for role, msg in st.session_state.chat_history_ui:
    with st.chat_message(role):
        st.write(msg)
