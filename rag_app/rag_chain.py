from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

import config

def create_rag_chain(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_store",
        collection_name="deepseek_rag_memory"
    )

    retriever = vectorstore.as_retriever()

    # ---- FIXED MEMORY --------
    history = ChatMessageHistory()
    memory = ConversationBufferWindowMemory(
        k=3,
        input_key="chat_history",
        return_messages=True,
        chat_memory=history
    )

    # ---- LLM -------------
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_API_BASE
    )

    # ---- Prompt ----------
    system_prompt = SystemMessagePromptTemplate.from_template(config.SYSTEM_PROMPT)
    human_prompt = HumanMessagePromptTemplate.from_template("{question}")

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    # ---- Conversational Retrieval Chain -----
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
