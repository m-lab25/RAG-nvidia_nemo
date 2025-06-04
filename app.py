import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
import tempfile
import time
import base64

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

import streamlit as st
import base64

st.set_page_config(page_title="Chat with your PDF", layout="wide")

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Encode logos
spearhead_logo_light_base64 = image_to_base64("image/work4-removebg-preview_light.png")
spearhead_logo_dark_base64 = image_to_base64("image/work4-removebg-preview.png")
nvidia_logo_light_base64 = image_to_base64("image/work2-removebg-preview_light.png")
nvidia_logo_dark_base64 = image_to_base64("image/work2-removebg-preview.png")

favicon_base64 = image_to_base64("image/favicon_file.ico")  # path to your favicon



# Inject favicon using base64
favicon_html = f"""
<link rel="icon" href="data:image/x-icon;base64,{favicon_base64}" type="image/x-icon">
"""

# Insert favicon and layout HTML
html_code = f"""
{favicon_html}
<style>
    .header-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
        border-bottom: 1px solid #ccc;
        flex-wrap: nowrap;
    }}

    .header-title {{
        flex: 1;
        text-align: center;
        font-size: 1.4em;
        font-weight: bold;
        padding: 0 10px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: black;
    }}

    .header-logo {{
        height: 100px;
        width: auto;
    }}

    .spearhead-light {{
        display: block;
    }}
    .spearhead-dark {{
        display: none;
    }}

    .nvidia-light {{
        display: block;
    }}
    .nvidia-dark {{
        display: none;
    }}

    @media (prefers-color-scheme: dark) {{
        .header-container {{
            border-bottom: 1px solid #444;
        }}
        .header-title {{
            color: white;
        }}
        .spearhead-light {{
            display: none;
        }}
        .spearhead-dark {{
            display: block;
        }}
        .nvidia-light {{
            display: none;
        }}
        .nvidia-dark {{
            display: block;
        }}

    }}
</style>

<div class="header-container">
    <img src="data:image/png;base64,{spearhead_logo_dark_base64}" alt="Spearhead Logo Dark" class="header-logo spearhead-light">
    <img src="data:image/png;base64,{spearhead_logo_light_base64}" alt="Spearhead Logo Light" class="header-logo spearhead-dark">
    <div class="header-title">NVIDIA NEMO PDF Assistant</div>
    <img src="data:image/png;base64,{nvidia_logo_dark_base64}" alt="NVIDIA Logo" class="header-logo nvidia-light">
    <img src="data:image/png;base64,{nvidia_logo_light_base64}" alt="NVIDIA Logo" class="header-logo nvidia-dark">
</div>
"""

# Render everything
st.markdown(html_code, unsafe_allow_html=True)


# Sidebar - File uploader
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Display uploaded filenames
if uploaded_files:
    st.sidebar.markdown("### Uploaded Files")
    for file in uploaded_files:
        st.sidebar.write(file.name)

# Initialize LLM
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Define prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the provided context only.
<context>
{context}
</context>
Question: {input}
""")

# File processing function
def process_uploaded_pdfs(files):
    documents = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        documents.extend(docs)

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    embeddings = NVIDIAEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store, split_docs

# Initialize vector store if files are uploaded
if uploaded_files and "vectors" not in st.session_state:
    with st.spinner("Processing uploaded PDFs..."):
        st.session_state.vectors, st.session_state.split_docs = process_uploaded_pdfs(uploaded_files)
    st.success("Vector Store is ready!")

# Question input
# Input box (just label removed since we created our own above)
user_question = st.text_input(label="Sample label", placeholder="Type your question here...",label_visibility='hidden')


# Handle QA
if user_question and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': user_question})
    response_time = time.process_time() - start_time

    st.subheader("🧠 Answer")
    st.write(response['answer'])

    st.markdown(f"*⏱️ Response time: {response_time:.2f} seconds*")

    with st.expander("📄 Source Documents"):
        for i, doc in enumerate(response["context"]):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content)
            st.write("---")
elif user_question:
    st.warning("⚠️ Please upload and embed documents first.")
