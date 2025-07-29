import os
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "legalbot"
DIMENSION = 384

def create_or_load_vectorstore(pdf_path="data/international_law_handbook.pdf", namespace="legalbot"):
    # Create Pinecone index if not exists
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
        )

    # Load and chunk PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Upsert documents into Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        index_name=INDEX_NAME,
        namespace=namespace
    )
    return vectorstore

def load_vectorstore(embedding, namespace="legalbot"):
    return PineconeVectorStore.from_existing_index(
        embedding=embedding,
        index_name=INDEX_NAME,
        namespace=namespace
    )
