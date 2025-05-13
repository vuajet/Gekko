import streamlit as st
# Page config must be first for mobile responsiveness
st.set_page_config(page_title="Hi! I'm Gekko - Your AI Insurance Chatbot", page_icon="🦎", layout="wide")

import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Initialize vector store and retriever
store = Chroma(
    persist_directory="./vectordb",
    embedding_function=OpenAIEmbeddings()
)
default_retriever = store.as_retriever(search_kwargs={"k": 5})

# Common conversational triggers
GREETINGS = {"hi", "hello", "hey"}
THANK_YOU = {"thank you", "thanks", "thankyou", "thx", "thank u"}

# Streamlit UI setup
st.title("🦎 Hi! I'm Gekko - Your AI Insurance Chatbot")
st.write("Ask me anything about GEICO’s insurance documentation across all lines of business.")
# Inject custom CSS for responsiveness on mobile
st.markdown(
    """
    <style>
    /* Expand container width */
    .appview-container .main .block-container { max-width: 95% !important; padding: 1rem !important; }
    /* Make text input full width */
    .stTextInput>div>div>input { width: 100% !important; }
    /* Style chat messages container */
    .chat-container { overflow-y: auto; max-height: 70vh; }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    send = st.form_submit_button("Send")

if send and user_input:
    user_clean = user_input.strip().lower()
    # Handle greetings
    if user_clean in GREETINGS:
        llm = ChatOpenAI(temperature=0)
        response = llm([HumanMessage(content=user_input)])
        answer = response.content
        st.session_state.history.insert(0, ("Assistant", answer))
        st.session_state.history.insert(0, ("You", user_input))
    # Handle thank you
    elif user_clean in THANK_YOU:
        answer = "You’re welcome! Let me know if there’s anything else I can help you with."
        st.session_state.history.insert(0, ("Assistant", answer))
        st.session_state.history.insert(0, ("You", user_input))
    else:
        # Retrieve relevant docs and sources
        docs = default_retriever.get_relevant_documents(user_input)
        sources = list({doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")})

        # Initialize LLM and memory
        llm = ChatOpenAI(temperature=0)
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,
            return_messages=True
        )

        # Build and run conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=default_retriever,
            memory=memory
        )
        with st.spinner("Thinking…"):
            result = qa_chain({"question": user_input})
            answer = result.get("answer", "")

        # Insert latest at top: You -> Assistant -> Source
        st.session_state.history.insert(0, ("Source", ", ".join(sources)))
        st.session_state.history.insert(0, ("Assistant", answer))
        st.session_state.history.insert(0, ("You", user_input))

# Display chat history (latest first) within responsive container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for speaker, message in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    elif speaker == "Assistant":
        st.markdown(f"**Assistant:** {message}")
    elif speaker == "Source":
        st.markdown(f"*Source:* {message}")
st.markdown('</div>', unsafe_allow_html=True)
