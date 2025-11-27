import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

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

    return loader.load()
