
# 🤖 PDF Chat Assistant – NeMo v1 (GraphRAG Ready)

A **Retrieval-Augmented Generation (RAG)** demo built with **NVIDIA NeMo**, enabling users to upload PDFs and interact with their content using natural language questions.

> ✅ Currently uses: FAISS-based retrieval  
> 🚧 Upcoming (v2): GraphRAG integration (graph-based contextual search)

---

## 🌐 Overview

This application allows users to:

- Upload one or more PDF files.
- Parse and chunk the text using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- Create semantic embeddings using `NVIDIAEmbeddings`.
- Store and search document chunks in a persistent `FAISS` vector index.
- Ask questions and retrieve relevant information grounded in context.
- Get answers using `meta/llama3-70b-instruct` via NVIDIA NeMo.

---

## 🧠 What Is RAG?

**Retrieval-Augmented Generation (RAG)** enhances large language models (LLMs) by combining them with an external knowledge base (e.g., PDF documents). Instead of generating from scratch, the LLM retrieves relevant content and answers based on that, which results in:

- Increased accuracy
- Lower hallucination
- Real-time document-based reasoning

---

## 🧰 Tech Stack

| Component                | Technology                    |
|--------------------------|-------------------------------|
| 🔗 LLM                   | `meta/llama3-70b-instruct` via `ChatNVIDIA` |
| 📚 Document Storage      | FAISS Vector Store            |
| 📄 PDF Ingestion         | LangChain + `PyPDFLoader`     |
| 🧠 Embedding Model       | `NVIDIAEmbeddings`            |
| 💬 Frontend              | Streamlit                     |
| 📊 Chunking Strategy     | `RecursiveCharacterTextSplitter` |
| 💾 Persistent Storage    | Local folder with FAISS       |

---

## 🚀 Upcoming: GraphRAG (v2)

We are actively working on integrating **GraphRAG**, which will replace FAISS with a graph-based retriever. This upgrade will enable:

- Better context awareness by linking semantically related chunks.
- Graph traversal algorithms for topic-centric exploration.
- Advanced reasoning through node-to-node relationships.

🛠️ *Stay tuned for a major update in the next release!*

---

## 🏗️ Architecture Diagram

```mermaid
flowchart TD
    A[📁 Upload PDFs] --> B[📄 PyPDFLoader]
    B --> C[🔗 Chunk via TextSplitter]
    C --> D[🧠 NVIDIA Embeddings]
    D --> E[🗂️ FAISS Vector Store Persistent data Base]

    F[❓ User Asks Question] --> G[🔍 FAISS Retriever]
    G --> H[📦 Prompt Assembly]
    H --> I[🤖 meta/llama3-70b-instruct]
    I --> J[💬 Streamlit Answer + Source Chunks]
````

---

## 🧬 System Workflow

### 1. 📁 Upload PDFs

PDFs are uploaded via the Streamlit sidebar and stored temporarily.

### 2. 📄 Document Parsing

`PyPDFLoader` loads the document and returns structured content.

### 3. 🔗 Text Chunking

Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`.

### 4. 🧠 Embedding

Chunks are embedded using `NVIDIAEmbeddings` to convert text into semantic vectors.

### 5. 🗂️ FAISS Index

Chunks are saved into FAISS (on-disk), enabling fast vector-based similarity search.

### 6. ❓ Question Answering

User questions are embedded and matched against stored vectors.

### 7. 🤖 LLM Generation

Retrieved context is passed into `meta/llama3-70b-instruct`, which generates a grounded answer.

### 8. 💬 Answer Display

Streamlit shows the answer along with document sources and latency.

---

## ✅ Features

* 💾 **Persistent Vector DB**: Upload once, reuse always.
* 🔍 **Context-Rich Answers**: Powered by semantic search.
* 📄 **Multi-Document Support**: Handle multiple PDFs at once.
* 🧠 **LLM Integration**: High-quality answers from Llama 3.
* 🔜 **GraphRAG (Next)**: Graph-based reasoning engine (coming soon).

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/nemo-pdf-chat-demo.git
cd nemo-pdf-chat-demo
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

---

## 🧪 Run the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 File Structure

```
.
├── app.py               # Main Streamlit App
├── vectorstore/         # Stored FAISS DBs (one per PDF)
├── image/               # Logos and favicon
├── .env                 # API Keys
├── requirements.txt     # Dependencies
└── README.md            # This file
```

---


📢 **Next Release:** Full GraphRAG integration with semantic graph traversal, node linking, and more!

```
