import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

# Load your API key
load_dotenv()

# 1. Re-open the persisted vector store
store = Chroma(
    persist_directory="./vectordb",
    embedding_function=OpenAIEmbeddings()
)
retriever = store.as_retriever(search_kwargs={"k": 3})

# 2. Build the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.5),
    chain_type="stuff",
    retriever=retriever
)

# 3. Simple interactive loop
print("🗨️  Geico Homeowners Assistant. Type your question (or 'exit' to quit).")
while True:
    query = input("You: ")
    if query.lower() in ("exit", "quit"):
        print("👋  Goodbye!")
        break
    answer = qa.run(query)
    print("\nAssistant:", answer, "\n")
