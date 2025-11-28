import streamlit as st
from loaders import load_document
from rag_chain import create_rag_chain
from utils import init_session

# Here I am setting up a Streamlit application for a RAG (Retrieval-Augmented Generation) chat bot. The app allows users to upload documents and interact with an AI assistant that answers questions based on the content of the uploaded documents.
st.set_page_config(page_title="RAG Chat Application", page_icon="💬")
st.title("RAG Chat Bot")

# Initialize session state variables (chat history and QA chain).
init_session()

# Here I am providing a button to clear the chat history and reset the QA chain.
if st.button("Clear Chat"):
    st.session_state.chat_history_ui = []
    st.session_state.qa_chain = None
    st.success("Chat cleared. Upload file again.")

# here I am providing a file uploader for users to upload documents in PDF, TXT, or DOCX formats.
uploaded_file = st.file_uploader("Upload PDF / TXT / DOCX")

# here I am processing the uploaded file to create a RAG chain using the provided documents.
if uploaded_file:
    docs = load_document(uploaded_file)
    print("docs_file_path :", docs)

    # Here I am storing loaded data into vector store and creating RAG chain and then storing that data into session state variable.
    st.session_state.qa_chain = create_rag_chain(docs)
    st.success("File processed. Start chatting!")

# Here I am providing a chat input box for users to ask questions to the AI assistant.
user_input = st.chat_input("Ask something...")

# Here I am handling user input, invoking the RAG chain to get responses as per user input, and updating the chat history.
if user_input and st.session_state.qa_chain:
    st.session_state.chat_history_ui.append(("user", user_input))

    # here I am displaying a spinner while the AI assistant is processing the user's question.
    with st.spinner("Thinking..."):

        # Here I am prividing static session id for chat history management.
        # Note: Instead of 'invoke' i can use 'stream' also for streaming response.
        response = st.session_state.qa_chain.invoke(
            {"question": user_input}, config={"configurable": {"session_id": "chat1"}}
        )

    # Here I am extracting the answer from the response and updating the chat history and storing in chat_history_ui session state variable to maintain the chat history.
    answer = response.content
    st.session_state.chat_history_ui.append(("assistant", answer))

# Here I am displaying the chat history in the Streamlit app.
for role, msg in st.session_state.chat_history_ui:
    with st.chat_message(role):
        st.write(msg)
