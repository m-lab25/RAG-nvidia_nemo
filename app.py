import streamlit as st
import os
import base64
import tempfile
import time
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# Load environment variables
from dotenv import load_dotenv
load_dotenv()


os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

st.set_page_config(page_title="Chat with your PDF", layout="wide")

VECTOR_ROOT = "vectorstore"

# -- Encode and render logos (unchanged section) --
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Encode logos
spearhead_logo_light_base64 = image_to_base64("image/work4-removebg-preview_light.png")
spearhead_logo_dark_base64 = image_to_base64("image/work4-removebg-preview.png")
nvidia_logo_light_base64 = image_to_base64("image/work2-removebg-preview_light.png")
nvidia_logo_dark_base64 = image_to_base64("image/work2-removebg-preview.png")
Hewlett_logo_light_base64 = image_to_base64("image/Hewlett_Packard_Enterprise-Logo.wine-removebg-preview.png")
Hewlett_logo_dark_base64 = image_to_base64("image/Hewlett_Packard_Enterprise-Logo.wine-removebg-preview.png")
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
    <img src="data:image/png;base64,{Hewlett_logo_light_base64}" alt="NVIDIA Logo" class="header-logo nvidia-light">
    <img src="data:image/png;base64,{Hewlett_logo_light_base64}" alt="NVIDIA Logo" class="header-logo nvidia-dark">
    <img src="data:image/png;base64,{nvidia_logo_dark_base64}" alt="NVIDIA Logo" class="header-logo nvidia-light">
    <img src="data:image/png;base64,{nvidia_logo_light_base64}" alt="NVIDIA Logo" class="header-logo nvidia-dark">
</div>
"""

# Render everything
st.markdown(html_code, unsafe_allow_html=True)


# -- Sidebar: Upload PDFs --
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Create directory if not exists
os.makedirs(VECTOR_ROOT, exist_ok=True)

# -- Process and store embeddings for each PDF --
import re

def sanitize_filename(name):
    # Remove non-alphanumeric characters and replace spaces with underscores
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

def process_pdf(file):
    raw_filename = os.path.splitext(file.name)[0]
    safe_filename = sanitize_filename(raw_filename)
    vector_path = os.path.join(VECTOR_ROOT, safe_filename)

    os.makedirs(vector_path, exist_ok=True)  # <-- ensure directory exists

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    embeddings = NVIDIAEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(vector_path)

    return safe_filename


from langchain_community.vectorstores import FAISS

def load_and_merge_all_vectorstores():
    embeddings = NVIDIAEmbeddings()
    merged_store = None

    for folder in pdf_folders:
        path = os.path.join(VECTOR_ROOT, folder)
        vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

        if merged_store is None:
            merged_store = vs
        else:
            merged_store.merge_from(vs)

    return merged_store.as_retriever()

# Process new uploads
if uploaded_files:
    for file in uploaded_files:
        fname = os.path.splitext(file.name)[0]
        target_path = os.path.join(VECTOR_ROOT, fname)
        if not os.path.exists(target_path):
            st.sidebar.write(f"📥 Indexing: {file.name}")
            process_pdf(file)
            st.sidebar.success(f"✅ Indexed: {file.name}")
        else:
            st.sidebar.info(f"ℹ️ Already exists: {file.name}")

# -- Sidebar: Show embedded PDFs --
# st.sidebar.markdown("### 📚 Available Embedded PDFs")

# for name in pdf_folders:
#     st.sidebar.markdown(f"- {name}.pdf")

st.sidebar.markdown("### 📚 Available Embedded PDFs")
pdf_folders = sorted([f for f in os.listdir(VECTOR_ROOT) if os.path.isdir(os.path.join(VECTOR_ROOT, f))])

for name in pdf_folders:
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    with col1:
        st.markdown(f"- {name}.pdf")
    with col2:
        if st.button(f"❌", key=f"remove_{name}"):
            path_to_delete = os.path.join(VECTOR_ROOT, name)
            try:
                # Remove vector store folder
                import shutil
                shutil.rmtree(path_to_delete)
                # st.rerun()# st.experimental_rerun()  # Refresh app to update sidebar
            except Exception as e:
                st.sidebar.error(f"Failed to remove {name}: {e}")

# -- Chat History --
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -- Display Chat History --
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# -- Chat Input --
user_question = st.chat_input("Ask something about your PDF...")

# -- Load all vectors for retrieval --
def load_all_vectors():
    embeddings = NVIDIAEmbeddings()
    vectorstores = []
    for folder in pdf_folders:
        path = os.path.join(VECTOR_ROOT, folder)
        vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        vectorstores.append(vs)
    return vectorstores

# -- Handle User Input --
if user_question:
    if not pdf_folders:
        st.warning("⚠️ Please upload and index at least one PDF.")
    else:
        # Load and merge all vector stores
        retriever = load_and_merge_all_vectorstores()
        # Now build the chain
        llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
        prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the provided context only.
        <context>
        {context}
        </context>
        Question: {input}
        """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Perform QA
        result = retrieval_chain.invoke({"input": user_question})


        start = time.process_time()
        result = retrieval_chain.invoke({"input": user_question})
        # result = chain.invoke({"input": user_question})
        end = time.process_time()

        st.session_state.chat_history.append((user_question, result["answer"]))

        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            st.markdown(f"*⏱️ {end - start:.2f}s*")

        with st.expander("📄 Source Documents"):
            for i, doc in enumerate(result["context"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
