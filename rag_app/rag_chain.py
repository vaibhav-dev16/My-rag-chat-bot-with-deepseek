from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI

import config


def create_rag_chain(docs):

    # --- Split documents ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    # --- Embeddings ---
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    # --- Vector DB ---
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_store",
        collection_name="deepseek_rag_memory",
    )
    retriever = vectorstore.as_retriever()

    # --- LLM ---
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_API_BASE,
        temperature=0,
    )

    # --- Prompt ---
    prompt = ChatPromptTemplate.from_messages(
        [("system", config.SYSTEM_PROMPT), ("human", "{question}")]
    )

    # --- RAG pipeline: Retrieve → Build context → LLM ---
    rag_pipeline = (
        {
            "context": lambda x: retriever.invoke(x["question"]),  # retrieves docs
            "question": RunnablePassthrough(),  # passes question untouched
        }
        | prompt
        | llm
    )

    # --- Memory wrapper ---
    def get_history(session_id: str):
        return ChatMessageHistory()

    rag_with_memory = RunnableWithMessageHistory(
        rag_pipeline,
        get_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return rag_with_memory
