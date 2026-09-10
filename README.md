# 📚 DocsMate

DocsMate is a **Retrieval-Augmented Generation (RAG)** document assistant that allows users to upload a PDF and ask questions about its content.

Instead of sending the entire document to the LLM, DocsMate retrieves the most relevant sections from the document and uses them as context to generate accurate, document-grounded answers.

---

## 🚀 Features

- 📄 Upload PDF documents
- ✂️ Split documents into smaller chunks
- 🔢 Generate semantic embeddings using Hugging Face
- 🗄️ Store document embeddings in ChromaDB
- 🔎 Retrieve relevant document sections using MMR
- 🤖 Generate answers using Groq's Qwen 3.6 27B
- 💬 Interactive chat interface using Streamlit
- 🧠 Maintains chat history during the session
- 🔐 Separate vector database for each user session
- 🚪 Clear chat or end the current session
- ❌ Prevents the model from answering outside the uploaded document

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **Streamlit** | Web interface |
| **LangChain** | RAG pipeline and LLM integration |
| **Hugging Face** | Text embeddings |
| **ChromaDB** | Vector database |
| **Groq** | LLM inference |
| **Qwen 3.6 27B** | Large Language Model |
| **PyPDF** | PDF document loading |
| **Pydantic** | Data validation |

---

## 📌 Future Improvements

- Support for multiple document formats
- Source/page references in answers
- Improved document management
- Persistent cloud vector storage
- Authentication and user accounts

---

## 👨‍💻 Author

**Purvesh Patil**

Built as a practical implementation of **Retrieval-Augmented Generation using LangChain, Hugging Face embeddings, ChromaDB, Streamlit, and Groq**.

---

## ⭐ Project Goal

DocsMate demonstrates how a modern RAG system can combine:

**Document Processing → Embeddings → Vector Search → Context Retrieval → LLM Generation**

to create a document-aware AI assistant that answers questions based on the user's uploaded documents.
