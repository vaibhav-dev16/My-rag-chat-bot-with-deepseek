import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


# Here I am defining a function to load documents based on the file type (PDF, TXT, DOCX) using appropriate loaders from langchain_community.
def load_document(file):
    ext = file.name.split(".")[-1]
    temp = f"temp.{ext}"

    with open(temp, "wb") as f:
        f.write(file.read())

    if ext == "pdf":
        loader = PyPDFLoader(temp)
    elif ext == "txt":
        loader = TextLoader(temp)
    elif ext in ["docx", "doc"]:
        loader = Docx2txtLoader(temp)
    else:
        st.error("Unsupported file format.")
        return None
    # Here .load() method loads and returns the documents from the file.
    return loader.load()
