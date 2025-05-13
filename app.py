import streamlit as st
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from load_data import load_and_chunk

# Page config for mobile responsiveness
st.set_page_config(page_title="Gecko Insurance Chatbot", page_icon="🦎", layout="wide")

# Load API key from Streamlit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Ingest and build vector store once
@st.cache_resource
def build_retriever():
    with st.spinner("Loading and embedding documents…"):
        docs = load_and_chunk()
        embeddings = OpenAIEmbeddings()
        store = FAISS.from_documents(docs, embeddings)
        return store.as_retriever(search_kwargs={"k": 5})

retriever = build_retriever()

# Conversation triggers
GREETINGS = {"hi", "hello", "hey"}
THANK_YOU = {"thank you", "thanks", "thankyou", "thx", "thank u"}

# Streamlit UI elements
st.title("🦎 Hi! I'm Gekko - Your AI Insurance Advisor")
st.write("Ask me anything about GEICO’s insurance documentation across all lines of business.")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    send = st.form_submit_button("Send")

# Handle interactions
if send and user_input:
    text = user_input.strip()
    key = text.lower()
    # Greetings
    if key in GREETINGS:
        response = ChatOpenAI(temperature=0)([HumanMessage(content=text)])
        answer = response.content
        st.session_state.history.insert(0, ("Assistant", answer))
        st.session_state.history.insert(0, ("You", text))
    # Thank you
    elif key in THANK_YOU:
        answer = "You’re welcome! Feel free to ask anything else about GEICO insurance."
        st.session_state.history.insert(0, ("Assistant", answer))
        st.session_state.history.insert(0, ("You", text))
    else:
        # Retrieve docs
        docs = retriever.get_relevant_documents(text)
        sources = list({doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")})

        # Build and run chain
        llm = ChatOpenAI(temperature=0)
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,
            return_messages=True
        )
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        with st.spinner("Thinking…"):
            result = qa({"question": text})
            answer = result.get("answer", "")

        # Insert in history: You -> Assistant -> Source
        st.session_state.history.insert(0, ("Source", ", ".join(sources)))
        st.session_state.history.insert(0, ("Assistant", answer))
        st.session_state.history.insert(0, ("You", text))

# Display chat history (latest first)
st.markdown("<div style='max-height:70vh; overflow:auto;'>", unsafe_allow_html=True)
for speaker, message in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    elif speaker == "Assistant":
        st.markdown(f"**Assistant:** {message}")
    elif speaker == "Source":
        st.markdown(f"*Source:* {message}")
st.markdown("</div>", unsafe_allow_html=True)
