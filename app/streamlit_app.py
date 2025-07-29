import streamlit as st
from app.vector_store import create_or_load_vectorstore, load_vectorstore
from app.chatbot import build_chatbot_chain
from langchain.embeddings import HuggingFaceEmbeddings

st.title("Legal Assistant Chatbot")

@st.cache_resource
def load_resources():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = create_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chain = build_chatbot_chain(retriever)
    return chain

chain = load_resources()

question = st.text_input("Ask a legal question:")

if question:
    with st.spinner("Generating answer..."):
        response = chain.invoke({"question": question})
        st.markdown(f"**Answer:** {response['answer']}")
