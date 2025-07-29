import os
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain

def build_chatbot_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a professional legal assistant. Answer ONLY based on the provided context. If the answer is not in the context, say 'I don't know.'"
        ),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\nQuestion: {question}"
        )
    ])

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain
