# Main application entry point
import streamlit as st
from dotenv import load_dotenv
from utils.loader import load_and_split_pdf
from utils.embedder import embed_chunks
from utils.qa import query_vectordb
import os

load_dotenv()

st.set_page_config(page_title="GenAI PDF Chatbot", page_icon="🤖")
st.title("🤖 Chat with our Assistant")

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Only process and embed once per session
if "pdf_embedded" not in st.session_state:
    with st.spinner("🔄 Reading & chunking..."):
        chunks = load_and_split_pdf("data/sample.pdf")
    with st.spinner("🧠 Embedding..."):
        embed_chunks(chunks)
    st.session_state["pdf_embedded"] = True

# Chat input
question = st.chat_input("Ask a question about your PDF...")

if question:
    with st.spinner("🤔 Thinking..."):
        answer = query_vectordb(question)

    # Store messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
