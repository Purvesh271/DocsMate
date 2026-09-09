import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import uuid
import shutil
import gc

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Config
load_dotenv()

st.set_page_config(
    page_title="DocsMate",
    page_icon="📚",
    layout="wide"
)

with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Database path
db_path = os.path.join( "chroma_db", st.session_state.session_id )

# Sidebar
with st.sidebar:

    st.title("📚 DocsMate")

    st.caption("Document Intelligence")

    st.divider()

    # Document
    st.subheader("📄 Document")

    st.caption( "Upload a PDF and create a private knowledge base." )

    uploaded_file = st.file_uploader( "Upload PDF", type=["pdf"] )

    # Uploaded file
    if uploaded_file:

        st.caption(f"📄 {uploaded_file.name}")

        if st.button( "⚡ Create Knowledge Base", type="primary", use_container_width=True ):
            with st.spinner( "Processing your document..." ):

                # Release previous Chroma instance
                st.session_state.vectorstore = None

                gc.collect()

                # Remove previous database
                if os.path.exists(db_path):
                    shutil.rmtree( db_path, ignore_errors=True )

                # Clear old chat
                st.session_state.messages = []

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile( delete=False, suffix=".pdf" ) as tmp_file:
                    
                    tmp_file.write( uploaded_file.getvalue() )

                    file_path = tmp_file.name

                try:

                    # Load PDF
                    loader = PyPDFLoader( file_path )

                    docs = loader.load()

                    # Split document
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    chunks = splitter.split_documents( docs )

                    # Embeddings
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )

                    # Chroma
                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=db_path
                    )

                    # Save database in session
                    st.session_state.vectorstore = vectorstore

                    st.session_state.pdf_name = ( uploaded_file.name )

                finally:

                    # Delete temporary PDF
                    if os.path.exists(file_path):
                        os.remove(file_path)

            st.caption("✓ Knowledge base ready")

            st.rerun()

    # Session
    st.divider()

    st.subheader("⚙️ Session")

    # Clear chat
    if st.button(  "🧹 Clear Chat", use_container_width=True ):
        st.session_state.messages = []

        st.rerun()

    # Quit session
    if st.button( "🚪 Quit Session", use_container_width=True ):

        # Release Chroma
        st.session_state.vectorstore = None

        gc.collect()

        # Delete session database
        if os.path.exists(db_path):
            shutil.rmtree( db_path, ignore_errors=True )

        # Clear session
        st.session_state.clear()

        st.success( "Session ended successfully." )

        st.stop()

# Main area
st.title("📚 DocsMate")

st.caption(  "Chat with your documents using Retrieval-Augmented Generation" )

# No document
if not st.session_state.vectorstore:

    st.divider()

    st.subheader(
        "Welcome to DocsMate 👋"
    )

    st.write(
        "Upload a PDF from the sidebar to start chatting "
        "with your document."
    )

    st.info(
        "💡 Your questions are answered using the content "
        "retrieved from your uploaded document."
    )

# Document ready
if st.session_state.vectorstore:

    st.divider()

    # Chat header
    col1, col2 = st.columns( [5, 1] )

    with col1:
        st.subheader( "💬 Document Chat" )
        st.caption(  st.session_state.pdf_name )

    with col2:
        st.success("Ready")

    # Retriever
    vectorstore = st.session_state.vectorstore

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # LLM
    llm = ChatGroq( model="qwen/qwen3.6-27b", temperature=0, max_tokens=800, reasoning_effort="none" )

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ( "system", """You are a helpful AI assistant.
                Use ONLY the provided context to answer the question.
                If the answer is not present in the context, 
                say: "I could not find the answer in the document." """
            ),
            ( "human", """Context: {context} Question: {question}""" )
        ]
    )

    # CHAT HISTORY
    for message in st.session_state.messages:

        with st.chat_message( message["role"] ):
            st.markdown( message["content"] )

    # CHAT INPUT
    query = st.chat_input( "Ask something about your document...")

    # PROCESS QUESTION
    if query:

        # User message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):

            st.markdown(query)

        # AI response
        with st.chat_message("assistant"):

            with st.spinner( "🔎 Searching your document..."):

                # Retrieve relevant chunks
                docs = retriever.invoke( query )

                # Create context
                context = "\n\n".join(
                    [
                        doc.page_content
                        for doc in docs
                    ]
                )

                # Create final prompt
                final_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": query
                    }
                )

                # Generate answer
                response = llm.invoke( final_prompt )

                answer = response.content

            st.markdown(answer)

        # Save AI response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )