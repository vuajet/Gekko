import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from load_data import load_and_chunk

# Load your OPENAI_API_KEY from .env
load_dotenv()

# 1. Load and chunk the Geico page content
docs = load_and_chunk()

# 2. Create embeddings model
emb = OpenAIEmbeddings()

# 3. Build (and persist) a local Chroma vector store
store = Chroma.from_documents(
    docs,
    embedding=emb,
    persist_directory="./vectordb"
)
store.persist()

print("✅ Vector store created with", store._collection.count(), "vectors.")
