from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI

import config


# Here I am creating a RAG (Retrieval-Augmented Generation) chain that processes documents, creates embeddings, stores them in a vector store, and sets up a chat-based LLM with memory to answer questions based on the documents.
def create_rag_chain(docs):

    # Here I am splitting the documents into smaller chunks for better processing and embedding.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    # Here I am creating embeddings for the document chunks.
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    # Here I am storing the embeddings in a Chroma vector store for efficient retrieval.
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_store",
        collection_name="deepseek_rag_memory",
    )
    retriever = vectorstore.as_retriever()  # create a retriever from the vector store

    # Here I am setting up a chat-based LLM (Language Model) using Deepseek's ChatOpenAI with a prompt template and integrating it with the retriever to form the RAG pipeline.
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_API_BASE,
        temperature=0,
    )

    # Here I am defining the prompt template for the chat-based LLM.
    prompt = ChatPromptTemplate.from_messages(
        [("system", config.SYSTEM_PROMPT), ("human", "{question}")]
    )

    # Here I am creating the RAG pipeline by combining the first getting retriever then pass that retriver to prompt and then pass that combined prompt to the llm
    rag_pipeline = (
        {
            "context": lambda x: retriever.invoke(x["question"]),  # retrieves docs
            "question": RunnablePassthrough(),  # passes question untouched
        }
        | prompt
        | llm
    )

    # Here I am adding message history management to the RAG pipeline to maintain chat history across interactions.
    def get_history(session_id: str):
        return ChatMessageHistory()

    # Here I am wrapping the RAG pipeline with message history functionality and returning the final RAG chain with memory.
    rag_with_memory = RunnableWithMessageHistory(
        rag_pipeline,
        get_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return rag_with_memory
